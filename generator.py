import torch.nn as nn
import torch
from torch.nn.modules import dropout


class G(nn.Module):
    def __init__(self, noise_size, output_size):
        super().__init__()
        
        hidden_size_lstm = 64
        hidden_size2 = 64
        hidden_size3 = 128
        

        self.lstm = nn.LSTM(
            input_size = noise_size,
            hidden_size = hidden_size_lstm,
            num_layers = 2,
            batch_first = True,
            dropout = 0.5,
            bidirectional = True)

        self.linear = nn.Sequential(
            nn.Linear(hidden_size_lstm*2, hidden_size2),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(hidden_size2, hidden_size3),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(hidden_size3, output_size),
            nn.Sigmoid()
        )

    # output shape (1, n_time, notes*instruments)
    # but needs a transposition
    def forward(self, noise):
        
        o, _ = self.lstm(noise)
        o = self.linear(o)
        
        return o


