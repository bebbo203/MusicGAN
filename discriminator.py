import torch.nn as nn
import torch
from torch.nn.modules import dropout
from dataset import MyDataset


class D(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        
        hidden_size = 32
        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = 2,
            batch_first = True,
            dropout = 0.4,
            bidirectional = True)

        self.linear = nn.Linear(hidden_size * 2, 2)

    # output shape (1, n_time, notes*instruments)
    # but needs a transposition
    def forward(self, x):
        
        _, (h, _) = self.lstm(x)
        h = h[2:, :, :].transpose(0,1).reshape(x.shape[0], -1)
        o = self.linear(h)
        return o

