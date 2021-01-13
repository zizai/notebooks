import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.policy import Policy
from ray.rllib.utils.annotations import override
from torch.nn.modules.transformer import Transformer

from models import DTM, GPT2Tokenizer, TransformerDiscriminator, VGAE
from utils import deep_update

logger = logging.getLogger(__name__)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


class ArwenModel(nn.Module):
    def __init__(self, device=torch.device("cpu"), mem_len=4096, z_dim=512, infow=1.):
        super(ArwenModel, self).__init__()
        self.device = device

        # model config
        self.mem_len = mem_len
        self.z_dim = z_dim

        # hyperparameters
        self.infow = infow
        self.delta_r = 1e-3
        self.goal_metric = 'l2'
        self.goal_threshold = 1e-3

        self.discriminator = TransformerDiscriminator(self.z_dim, n_heads=8, n_layers=3)

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.embed = nn.Embedding(self.tokenizer.vocab_size, self.z_dim)
        self.out = nn.Linear(self.z_dim, self.tokenizer.vocab_size)
        self.out.weight = self.embed.weight

        self.transformer = Transformer(d_model=self.z_dim, num_encoder_layers=3, num_decoder_layers=3)
        self.vae = VGAE(self.z_dim, num_encoder_layers=6, num_decoder_layers=6)
        self.memory = DTM(self.mem_len, self.z_dim)

        self.generator = nn.ModuleList([self.embed, self.memory, self.transformer, self.vae])

    def p_z(self, y=None):
        if y is None:
            y = self.memory.p_y.sample()
        r = self.memory.read(y)
        p_z = self.vae.reparameterize(r)
        return p_z

    def q_z(self, y, r):
        self.memory.write(y, r)
        q_z = self.vae.reparameterize(r)
        return q_z

    def embed_input(self, text):
        tokens = self.tokenizer.encode(text)
        tokens = torch.tensor(tokens, device=self.device, dtype=torch.long).unsqueeze(-1)
        x = self.embed(tokens)
        return x

    def random_walk(self, y):
        r = self.memory.read(y)
        r_next = r + self.delta_r * torch.randn_like(r)
        y_next = self.memory.q_y(r_next).sample()
        return y_next

    def reach_goal(self, z, z_goal):
        if self.goal_metric == 'l1':
            distance = F.pairwise_distance(z, z_goal, p=1)
        else:
            distance = F.pairwise_distance(z, z_goal, p=2)
        if distance < self.goal_threshold:
            return True
        else:
            return False

    def plan(self, y, y_goal, max_len=10):
        y_next = self.memory.shortest_path(y[-1].unsqueeze(0), y_goal)
        if y_next is None and y.size(0) >= max_len:
            y_plan = torch.cat([y, self.random_walk(y)], dim=0)
        elif y_next is None and y.size(0) < max_len:
            y_next = torch.cat([y, self.random_walk(y)], dim=0)
            y_plan = self.plan(y_next, y_goal)
        elif y.size(0) > 1:
            y_plan = torch.cat([y[:-1].unsqueeze(0), y_next], dim=0)
        else:
            y_plan = y_next
        return y_plan

    def x_to_tokens(self, x):
        tokens = F.gumbel_softmax(self.out(x), hard=True)
        tokens = torch.argmax(tokens, dim=-1)
        if tokens.dim() > 1:
            tokens = tokens.squeeze()
        return tokens.tolist()

    def encode(self, x):
        x = self.transformer.encoder(x)  # [seq_len, 1, z_dim]
        r, _ = self.vae.encode(x.squeeze(1))  # [seq_len, z_dim]
        return r.sum(0, keepdim=True)  # [1, z_dim]

    def decode(self, y, y_goal):
        z = self.p_z(y).rsample()  # [1, z_dim]
        x_begin, _ = self.vae.decode(z)  # [seq_len, z_dim]
        x_begin = self.transformer.decoder(x_begin.unsqueeze(1), z)  # [seq_len, 1, z_dim]

        y_plan = self.plan(y, y_goal)
        z_plan = self.p_z(y_plan).rsample()

        x_end, _ = self.vae.decode(z_plan)
        x_end = self.transformer.decoder(x_end.unsqueeze(1), z_plan)

        tokens = self.x_to_tokens(x_end)
        text = self.tokenizer.decode(tokens)
        return text, x_begin, x_end, z, z_plan

    def sample_sequence(self, prompt):
        x = self.embed_input([prompt])
        r = self.encode(x)
        y = self.memory.q_y(r).sample()
        y_goal = self.memory.p_y.sample()
        results = self.decode(y, y_goal)
        return results[0]

    def bce_loss(self, logits, real):
        if real:
            real_labels = torch.ones_like(logits)
            bce_loss = F.binary_cross_entropy(logits, real_labels)
        else:
            fake_labels = torch.zeros_like(logits)
            bce_loss = F.binary_cross_entropy(logits, fake_labels)
        return bce_loss

    def mi_loss(self, x, y, z):
        ent_loss = - self.p_z(y).log_prob(z).mean(-1)
        r = self.encode(x)
        crossent_loss = - self.q_z(y, r).log_prob(z).mean(-1)
        return crossent_loss - ent_loss

    def discriminator_step(self, current, goal):
        x = self.embed_input(current)  # [seq_len, 1, z_dim]
        x_goal = self.embed_input(goal)

        # make inference
        r = self.encode(x)  # [1, z_dim]
        r_goal = self.encode(x_goal)  # [1, z_dim]

        # sample posterior
        y_hat = self.memory.q_y(r).sample()  # [1]
        y_goal_hat = self.memory.q_y(r).sample()  # [1]
        z_hat = self.q_z(y_hat, r).rsample()  # [1, z_dim]
        z_goal_hat = self.q_z(y_goal_hat, r_goal).rsample()  # [1, z_dim]

        # real samples
        real_logits = [self.discriminator(x, z_hat), self.discriminator(x_goal, z_goal_hat)]
        real_logits = torch.cat(real_logits, dim=0)
        real_loss = self.bce_loss(real_logits, True)

        # fake samples
        y = self.memory.p_y.sample()
        y_goal = self.memory.p_y.sample()
        text, x_begin, x_end, z, z_plan = self.decode(y, y_goal)
        z_goal = z_plan[-1].unsqueeze(0)

        fake_logits = [self.discriminator(x_begin.detach(), z), self.discriminator(x_end.detach(), z_goal)]
        fake_logits = torch.cat(fake_logits, dim=0)
        fake_loss = self.bce_loss(fake_logits, False)

        d_loss = real_loss + fake_loss
        return text, x_begin, x_end, y, y_goal, z, z_goal, d_loss

    def generator_step(self, x_begin, x_end, y, y_goal, z, z_goal):
        fake_logits = [self.discriminator(x_begin, z), self.discriminator(x_end, z_goal)]
        fake_logits = torch.cat(fake_logits, dim=-1)
        g_loss = self.bce_loss(fake_logits, True)

        mi_loss = self.mi_loss(x_begin, y, z)
        mi_loss_goal = self.mi_loss(x_end, y_goal, z_goal)
        q_loss = mi_loss + mi_loss_goal

        g_loss += self.infow * q_loss.squeeze()

        return g_loss


DEFAULT_CONFIG = {
    'model': {
        'mem_len': 4096,
        'z_dim': 512
    }
}


class ArwenPolicy(Policy):

    _default_config = DEFAULT_CONFIG

    def __init__(self, config):
        self.config = deep_update(self._default_config, config)
        self.device = (torch.device("cuda")
                       if bool(os.environ.get("CUDA_VISIBLE_DEVICES", None))
                       else torch.device("cpu"))

        model_config = config['model']
        mem_len = model_config['mem_len']
        z_dim = model_config['z_dim']
        self.model = ArwenModel(device=self.device, mem_len=mem_len, z_dim=z_dim)
        self.discriminator = self.model.discriminator
        self.generator = self.model.generator
        self.num_params = sum(p.numel() for p in self.model.parameters())

        self.model.requires_grad_(False)

    @override(Policy)
    def compute_single_action(self, obs, state=None):
        current, goal = obs.current, obs.goal
        text, x_begin, x_end, y, y_goal, z, z_goal, d_loss = self.model.discriminator_step(current, goal)
        g_loss = self.model.generator_step(x_begin, x_end, y, y_goal, z, z_goal)
        return text, d_loss, g_loss

    @override(Policy)
    def get_weights(self, type=None):
        if type == 'discriminator':
            params = self.discriminator.parameters()
        elif type == 'generator':
            params = self.generator.parameters()
        else:
            params = self.model.parameters()
        return nn.utils.parameters_to_vector(params).numpy()

    @override(Policy)
    def set_weights(self, weights, type=None):
        weights = torch.tensor(weights)
        if type == 'discriminator':
            params = self.discriminator.parameters()
        elif type == 'generator':
            params = self.generator.parameters()
        else:
            params = self.model.parameters()
        nn.utils.vector_to_parameters(weights, params)

    def import_model(self, import_dir):
        assert os.path.exists(import_dir)
        checkpoint = torch.load(import_dir)
        self.model.load_state_dict(checkpoint['model'])

    def export_model(self, export_dir):
        assert os.path.exists(export_dir)
        checkpoint = {
            'model': self.model.state_dict()
        }
        torch.save(checkpoint, export_dir)

    def export_checkpoint(self, export_dir):
        pass
