import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


from .core import Agent
from constants import MAX_SEQ_LEN
from models import RAPL, GPT2Tokenizer


class Librarian(Agent):
    def __init__(self, device, env, encoder_type='CNN', decoder_type='RNN', x_dim=3, y_dim=1, h_dim=512, s_dim=128, lr=3e-4):
        self.device = device
        self.env = env
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.s_dim = s_dim
        self.a_dim = env.action_space.shape
        self.lr = lr
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.vocab_size = self.tokenizer.vocab_size

        self.rapl = RAPL(device, self.vocab_size, x_dim, s_dim, self.a_dim,
                         h_dim=h_dim, encoder_type=encoder_type, decoder_type=decoder_type).to(self.device)
        self.optimizer = torch.optim.Adam(self.rapl.parameters(), lr=self.lr)
        self.action = None
        self.obs = None
        self.s_0 = torch.zeros(1, self.s_dim, device=self.device, dtype=torch.float)
        self.a_0 = torch.zeros(1, self.a_dim, device=self.device, dtype=torch.float)
        self.s_t = self.s_0
        self.a_t = self.a_0
        self.q_s = None
        self.p_s = None
        self.y_t = None
        self.y_t_pred = None
        self.fe_history = []

    @property
    def training(self):
        return self.rapl.training

    def _unpack_obs(self, obs):
        """

        :param obs: list
            key-value sequences

        :return k: torch.Tensor

        """
        k = []
        v = []
        for each_obs in obs:
            try:
                o = torch.tensor(each_obs)
                k_b = o[:MAX_SEQ_LEN, :self.x_dim]
                v_b = o[:MAX_SEQ_LEN, self.x_dim:].squeeze(1)
                k.append(k_b)
                v.append(v_b)
            except IndexError:
                print(each_obs)
                raise
        v_len = [line.shape[0] for line in v]
        v_len = torch.tensor(v_len, device=self.device, dtype=torch.long)

        k = pad_sequence(k, batch_first=True, padding_value=0).to(device=self.device, dtype=torch.float)
        v = pad_sequence(v, batch_first=True, padding_value=0).to(device=self.device, dtype=torch.long)

        return k, v, v_len

    def _print_status(self, t, fe):
        print(">>>>>>> timestep {}, free energy {:.3f}, batch size {} <<<<<<<".format(t, fe.item(), self.y_t.shape[0]))
        print("[input]")
        for i in self.y_t[:2, :].tolist():
            obs = self.tokenizer.decode(i)
            print(obs)

        print("[output]")
        for i in self.y_t_pred[:2, :].topk(1, dim=-1)[1].squeeze(-1).tolist():
            obs_pred = self.tokenizer.decode(i)
            print(obs_pred)

        return

    def loss(self, q_s, p_s, v, v_pred, v_len, ctc_loss=True):
        """
        Computes RAPL Free Energy.

        Parameters
        ----------
        q_s : torch.distributions.Distribution
            Variational distribution over state.
        p_s : torch.distributions.Distribution
            Prior distribution over state.
        v : torch.Tensor
            Shape (batch_size, seq_len)
        v_pred : torch.Tensor
            Shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len, _ = v_pred.size()
        if ctc_loss:
            log_probs = v_pred.transpose(0, 1).log_softmax(2)
            input_len = torch.full((batch_size,), seq_len, device=self.device, dtype=torch.long)
            rec_loss = F.ctc_loss(log_probs, v, input_len, v_len)
        else:
            rec_loss = 0
            for b in range(batch_size):
                v_batch = v[b, :]
                v_pred_batch = v_pred[b, :, :]
                rec_loss += F.cross_entropy(v_pred_batch, v_batch, reduction='sum')
            # average over batch, sum across sequence
            rec_loss = rec_loss / batch_size

        # average over batch, sum across dimensions
        kl = self._kl_divergence(q_s, p_s).mean(dim=0).sum()

        print('Reconstruction Loss: {}, KL Divergence: {}'.format(rec_loss.item(), kl.item()))
        return rec_loss + kl

    def take_action(self, obs):

        k, v, v_len = self._unpack_obs(obs)
        s_t, a_tp1, q_s, v_pred = self.rapl(k, v, v_len)

        self.action = torch.argmax(a_tp1).item()
        self.y_t = v
        self.y_t_pred = v_pred

        fe = self.loss(q_s, self.p_s, v, v_pred, v_len)

        return self.action, fe

    def train(self, n_episodes, n_steps):
        self.rapl.training = True
        for i_episode in range(n_episodes):
            self.action = self.env.action_space.sample()
            self.s_t = self.s_0
            self.a_t = self.a_0
            t = 0

            for t in range(n_steps):
                obs = self.env.step(self.action)
                _, fe = self.take_action(obs)

                self._print_status(t, fe)

                fe.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.fe_history.append(fe)

            avg_fe = torch.tensor(self.fe_history).mean(dim=0).item()
            print("episode: {}, total timesteps: {}, avg free energy {:.3f}".format(i_episode+1, t+1, avg_fe))
            print("=================================================")

            del self.fe_history[:]

        self.env.close()
        self.rapl.training = False
