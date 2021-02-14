import torch
import torch.nn as nn
from configuration import *


class GeneraterBlock(nn.Module):

    def __init__(self, in_dim, out_dim, kernel, stride):
        super().__init__()
        self.transconv = nn.ConvTranspose3d(in_dim, out_dim, kernel, stride)
        self.batchnorm = nn.BatchNorm3d(out_dim)
    
    def forward(self, x):
        x = self.transconv(x)
        x = self.batchnorm(x)
        return nn.functional.relu(x)

class G(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.transconv0 = GeneraterBlock(LATENT_DIM, 256, kernel=(4, 1, 1), stride=(4, 1, 1))
        self.transconv1 = GeneraterBlock(256, 128, kernel=(1, 4, 1), stride=(1, 4, 1))
        self.transconv2 = GeneraterBlock(128, 64, kernel=(1, 1, 4), stride=(1, 1, 4))
        self.transconv3 = GeneraterBlock(64, 32, kernel=(1, 1, 3), stride=(1, 1, 1))
        # a different layer for each track from now on
        self.transconv4 = nn.ModuleList([
            GeneraterBlock(32, 16, kernel=(1, 4, 1), stride=(1, 4, 1))
            for _ in range(N_TRACKS)
        ])
        self.transconv5 = nn.ModuleList([
            GeneraterBlock(16, 1, kernel=(1, 1, 12), stride=(1, 1, 12))
            for _ in range(N_TRACKS)
        ])

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, LATENT_DIM, 1, 1, 1)
        # common transpose convolutional layers
        x = self.transconv0(x)
        x = self.transconv1(x)
        x = self.transconv2(x)
        x = self.transconv3(x)
        # a different layer for each track
        x_list = [transconv(x) for transconv in self.transconv4]
        x_list = [transconv(x_) for x_, transconv in zip(x_list, self.transconv5)]
        # merge the branches' results concatenating them
        x = torch.cat(x_list, dim=1)
        x = x.view(batch_size, N_TRACKS, N_MEASURES_FOR_SAMPLE * MEASURE_LENGTH, N_PITCHES)
        return x

