import torch.nn as nn
import torch
from torch.nn.modules import dropout


class G(nn.Module):
    def __init__(self, noise_size, output_size):
        super().__init__()
        
        hidden_size = 32
        self.lstm = nn.LSTM(
            input_size = noise_size,
            hidden_size = hidden_size,
            num_layers = 2,
            batch_first = True,
            dropout = 0.4,
            bidirectional = False)

        self.linear = nn.Linear(hidden_size, output_size)

    # output shape (1, n_time, notes*instruments)
    # but needs a transposition
    def forward(self, noise):
        
        o, _ = self.lstm(noise)
        o = self.linear(o)
        
        return o


