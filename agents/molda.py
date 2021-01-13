import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib import Policy
from ray.rllib.utils.annotations import override

from datasets.molecular_dataset import MolecularDataset
from models import (GraphDiscriminator, VGAE, DTM)
from torch_geometric.data import (Data)
from utils import deep_update

'''
Graph Convolution Reinforcement Learning for Multi-Agent Cooperation
https://github.com/PKU-AI-Edge/DGN
'''


class MoldaModel(nn.Module):
    def __init__(self, x_dim, mem_len=1000, z_dim=256, device=torch.device("cpu")):
        super(MoldaModel, self).__init__()
        self.device = device

        # model config
        self.mem_len = mem_len
        self.z_dim = z_dim

        # hyperparameters
        self.infow = 1.
        self.delta_r = 1e-3
        self.goal_metric = 'l2'
        self.goal_threshold = 1e-3

        self.discriminator = GraphDiscriminator(x_dim, z_dim)

        self.embed = nn.Linear(x_dim, z_dim, bias=False)
        self.out = nn.Linear(z_dim, x_dim, bias=False)

        self.vae = VGAE(self.z_dim, num_encoder_layers=6, num_decoder_layers=6)
        self.memory = DTM(self.mem_len, self.z_dim)

        self.generator = nn.ModuleList([self.embed, self.memory, self.vae, self.out])

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

    def random_walk(self, y):
        r = self.memory.read(y)
        r_next = r + self.delta_r * torch.randn_like(r)
        y_next = self.memory.q_y(r_next).sample()
        return y_next

    def encode(self, x, e):
        x = self.embed(x)
        r, e = self.vae.encode(x, e)  # [x_len, z_dim]
        return r, e  # [r_len, z_dim]

    def decode(self, y):
        z = self.p_z(y).rsample()  # [1, z_dim]
        x, e = self.vae.decode(z)  # [seq_len, z_dim]
        x = self.out(x)
        return x, e, z

    def sample_results(self, g=None):
        if g is None:
            y = self.memory.p_y.sample()
        else:
            r = self.encode(g.x, g.edge_index)
            y = self.memory.q_y(r).sample()
        results = self.decode(y)
        return results

    def bce_loss(self, logits, real):
        if real:
            real_labels = torch.ones_like(logits)
            bce_loss = F.binary_cross_entropy(logits, real_labels)
        else:
            fake_labels = torch.zeros_like(logits)
            bce_loss = F.binary_cross_entropy(logits, fake_labels)
        return bce_loss

    def mi_loss(self, x, e, y, z):
        ent_loss = - self.p_z(y).log_prob(z).mean(-1)
        r, _ = self.encode(x, e)
        r = r.sum(0, keepdim=True)
        crossent_loss = - self.q_z(y, r).log_prob(z).mean(-1)
        return crossent_loss - ent_loss

    def discriminator_step(self, g: Data):
        x = g.x.to(device=self.device, dtype=torch.float)
        e = g.edge_index.to(self.device, dtype=torch.long)

        # make inference
        r, _ = self.encode(x, e)  # [1, z_dim]
        r = r.sum(0, keepdim=True)

        # sample posterior
        y_hat = self.memory.q_y(r).sample()  # [1]
        z_hat = self.q_z(y_hat, r).rsample()  # [1, z_dim]

        # real samples
        real_logits = self.discriminator(x, e, z_hat)
        real_loss = self.bce_loss(real_logits, True)

        # fake samples
        y_fake = self.memory.p_y.sample()
        x_fake, e_fake, z_fake = self.decode(y_fake)

        fake_logits = self.discriminator(x_fake, e_fake, z_fake)
        fake_loss = self.bce_loss(fake_logits, False)

        d_loss = real_loss + fake_loss
        return x_fake, e_fake, y_fake, z_fake, d_loss

    def generator_step(self, x, e, y, z):
        fake_logits = self.discriminator(x, e, z)
        g_loss = self.bce_loss(fake_logits, True)

        mi_loss = self.mi_loss(x, e.detach(), y, z)

        g_loss += self.infow * mi_loss.squeeze()
        return g_loss


DEFAULT_CONFIG = {
    'max_atom_num': MolecularDataset.max_atom_num,
    'model': {
        'mem_len': 4096,
        'z_dim': 512
    }
}


class MoldaPolicy(Policy):

    _default_config = DEFAULT_CONFIG

    def __init__(self, config):
        self.config = deep_update(self._default_config, config)
        self.device = (torch.device("cuda")
                       if bool(os.environ.get("CUDA_VISIBLE_DEVICES", None))
                       else torch.device("cpu"))

        # maximal atomic number is the x_dim of model
        self.max_atom_num = self.config['max_atom_num']

        self.model = MoldaModel(self.max_atom_num, **self.config['model'])
        self.discriminator = self.model.discriminator
        self.generator = self.model.generator
        self.num_params = sum(p.numel() for p in self.model.parameters())

        self.model.requires_grad_(False)

    def compute_gradients(self, postprocessed_batch):
        pass

    def apply_gradients(self, gradients):
        pass

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

    def compute_actions(self,
                        obs_batch,
                        state_batches,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        pass

    @override(Policy)
    def compute_single_action(self, obs, state=None):
        g = obs
        x_fake, e_fake, y_fake, z_fake, d_loss = self.model.discriminator_step(g)
        g_loss = self.model.generator_step(x_fake, e_fake, y_fake, z_fake)
        g = Data(x=x_fake, edge_index=e_fake)
        return g, d_loss, g_loss
