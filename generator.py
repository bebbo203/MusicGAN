import torch.nn as nn
import torch
from torch.nn.modules import dropout


class G(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        
        hidden_size = 32
        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = 2,
            batch_first = True,
            dropout = 0.4,
            bidirectional = False)

        self.linear = nn.Linear(hidden_size, 128)


    def forward(self, noise):
        
        o, _ = self.lstm(noise)

        print("o", o.shape)

        o = self.linear(o)

        print("l", o.shape)

        return o


g = G(96)
noise = torch.randn((1, 128*4, 96))
g(noise)

