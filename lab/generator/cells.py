import torch
import torch.nn as nn


class RNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def init_state(self, batch_size, device):
        return (torch.zeros(batch_size, self.hidden_dim, device=device),)

    def forward(self, x, state):
        (h,) = state
        h = torch.tanh(self.linear(torch.cat([x, h], dim=1)))
        return h, (h,), {}


class GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gates = nn.Linear(input_dim + hidden_dim, 2 * hidden_dim)
        self.x_to_n = nn.Linear(input_dim, hidden_dim)
        self.h_to_n = nn.Linear(hidden_dim, hidden_dim)

    def init_state(self, batch_size, device):
        return (torch.zeros(batch_size, self.hidden_dim, device=device),)

    def forward(self, x, state):
        (h,) = state
        z, r = torch.sigmoid(self.gates(torch.cat([x, h], dim=1))).chunk(2, dim=1)
        n = torch.tanh(self.x_to_n(x) + r * self.h_to_n(h))
        h = (1 - z) * h + z * n
        return h, (h,), {"update": z, "reset": r}


class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)
        with torch.no_grad():
            self.linear.bias[hidden_dim:2 * hidden_dim].fill_(1.0)

    def init_state(self, batch_size, device):
        zeros = torch.zeros(batch_size, self.hidden_dim, device=device)
        return (zeros, zeros.clone())

    def forward(self, x, state):
        h, c = state
        i, f, g, o = self.linear(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, (h, c), {"input": i, "forget": f, "output": o, "cell": c}


CELLS = {"rnn": RNNCell, "gru": GRUCell, "lstm": LSTMCell}
