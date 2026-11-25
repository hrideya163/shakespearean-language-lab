"""Recurrent cells written by hand, so we can see (and record) every gate.

All three cells have the same interface:
    h, new_state, gates = cell(x, state)
      x      (B, input_dim)  input at this time step
      state  tuple of tensors carried to the next step
      h      (B, hidden_dim) output of this step
      gates  dict of named tensors we want to inspect (empty for the plain RNN)

PyTorch's nn.LSTM is faster, but it hides the gates. Writing the equations
ourselves is the whole point of this project's "inside the LSTM" experiments.
"""
import torch
import torch.nn as nn


class RNNCell(nn.Module):
    """Vanilla RNN:  h_t = tanh(W [x_t, h_{t-1}] + b)
    No gates, so information from far back fades (the vanishing gradient problem)."""

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
    """GRU: two gates.
        z = sigmoid(...)   update gate: how much of the old state to replace
        r = sigmoid(...)   reset gate:  how much of the old state to use for the new candidate
        n = tanh(W x + r * (U h))        candidate
        h = (1 - z) * h + z * n
    """

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
    """LSTM: a separate memory (cell state c) protected by three gates.
        i = sigmoid(...)   input gate:  how much new information to write
        f = sigmoid(...)   forget gate: how much old memory to keep
        o = sigmoid(...)   output gate: how much memory to show as output
        g = tanh(...)      candidate new information
        c = f * c + i * g
        h = o * tanh(c)
    Because c is updated by multiplication with f (not squashed by tanh every step),
    gradients can flow back through many steps. This is why LSTMs remember longer than RNNs.
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)  # all four parts at once
        # Trick: start the forget gate "open" (bias 1) so the model remembers by default.
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
