import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from agents.core import Agent
from agents.utils import (convert_seq_to_graph, unpack_kv)
from models import (GraphDiscriminator, GraphEncoder, GraphDecoder, GPT2Tokenizer)


class ALI(Agent):
    def __init__(self, device, env, k_dim=3, v_dim=1, h_dim=512, s_dim=128, lr=3e-4):
        self.device = device
        self.env = env
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.h_dim = h_dim
        self.s_dim = s_dim
        self.a_dim = env.action_space.shape
        self.lr = lr
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.vocab_size = self.tokenizer.vocab_size

        self.embed = nn.Embedding(self.vocab_size, h_dim, padding_idx=0).to(device)
        self.out = nn.Linear(h_dim, self.vocab_size).to(device)

        self.discriminator = GraphDiscriminator(h_dim).to(device)
        self.encoder = GraphEncoder(h_dim).to(device)
        self.generator = GraphDecoder(h_dim).to(device)
        self.musigma = nn.Linear(h_dim, 2 * h_dim).to(device)

        # Tying weights in the input and output layer might increase performance (Inan et al., 2016)
        self.out.weight = self.embed.weight

        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        self.g_optimizer = torch.optim.Adam([
            {'params': self.embed.parameters()},
            {'params': self.encoder.parameters()},
            {'params': self.musigma.parameters()},
            {'params': self.generator.parameters()}], lr=lr)

        self.g_t = None
        self.g_sampler = None
        self.training = False

    def reparameterize(self, r):
        mu, sigma = torch.chunk(self.musigma(r), 2, dim=-1)
        sigma = 1e-6 + F.softplus(sigma)
        p_z = Normal(mu, sigma)
        return p_z

    def train_step(self, graphs):
        d_loss = []
        g_loss = []

        for g in graphs:
            v_t = g.x.to(self.device)
            e_t = g.edge_index.to(self.device)

            print("v_t size: {}, e_t size: {}".format(v_t.shape, e_t.shape))
            v_t_embedded = self.embed(v_t)

            r, edge_index, rs, edge_indices, perms = self.encoder(v_t_embedded, e_t)
            print("r size: {}".format(r.shape))
            q_z = self.reparameterize(r)
            z_tilde = q_z.rsample()

            p_z = Normal(torch.zeros_like(z_tilde), 1)
            z = p_z.rsample()
            v_tilde, _ = self.generator(z, edge_index, rs, edge_indices, perms)
            e_tilde = e_t

            fake_logits = self.discriminator(v_tilde, e_tilde, z)
            real_logits = self.discriminator(v_t_embedded, e_t, z_tilde)

            d_loss = F.binary_cross_entropy(real_logits, torch.ones_like(real_logits)) + \
                     F.binary_cross_entropy(fake_logits, torch.zeros_like(fake_logits))

            self.d_optimizer.zero_grad()
            d_loss.backward(retain_graph=True)
            self.d_optimizer.step()

            g_loss = F.binary_cross_entropy(real_logits, torch.zeros_like(real_logits)) + \
                     F.binary_cross_entropy(fake_logits, torch.ones_like(fake_logits))

            self.g_optimizer.zero_grad()
            g_loss.backward()
            self.g_optimizer.step()

        return d_loss, g_loss

    def train(self, n_episodes, n_steps, data_loader=None):
        self.training = True
        self.discriminator.train(True)
        self.encoder.train(True)
        self.generator.train(True)

        for i_episode in range(n_episodes):
            loss_history = []
            t = 0

            if data_loader is not None:
                for batch in data_loader:
                    d_loss, g_loss = self.train_step([batch])
                    print(">>>>>>> timestep {}, d_loss {:.3f}, g_loss {:.3f} <<<<<<<".format(
                        t, d_loss.item(), g_loss.item()))
                    t += 1
            else:
                for t in range(n_steps):
                    a_t = self.env.action_space.sample()
                    o_t = self.env.step(a_t)

                    k_t, v_t, v_len = unpack_kv(o_t, self.k_dim)
                    graphs = convert_seq_to_graph(v_t, max_graph_size=512, max_node_degree=16)
                    d_loss, g_loss = self.train_step(graphs)
                    loss_history.append(d_loss + g_loss)

                    print(">>>>>>> timestep {}, d_loss {:.3f}, g_loss {:.3f} <<<<<<<".format(
                        t, d_loss.item(), g_loss.item()))

            avg_loss = torch.tensor(loss_history).mean(dim=0).item()
            print("episode: {}, total timesteps: {}, avg loss {:.3f}".format(i_episode+1, t+1, avg_loss))
            print("=================================================")

            del loss_history[:]

        self.training = False
        self.discriminator.train(False)
        self.encoder.train(False)
        self.generator.train(False)

    def _print_status(self, v_t, v_t_tilde):
        print("[input]")
        obs = self.tokenizer.decode(v_t[:10].tolist())
        print(obs)

        print("[output]")
        obs_pred = self.tokenizer.decode(v_t_tilde[:10].topk(1, dim=-1)[1].squeeze(-1).tolist())
        print(obs_pred)


if __name__=='__main__':
    import os
    from chaosbreaker.envs import WikipediaLibrary

    data_dir = os.path.abspath('./data/wiki_test')
    wiki_env = WikipediaLibrary(data_dir)

    agent = ALI('cpu', wiki_env, s_dim=32, h_dim=64)
    agent.train(1, 3)
