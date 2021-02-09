import torch
import torch.nn as nn
from configuration import *



class DiscriminatorBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride, feature_shape):
        super().__init__()
        self.transconv = nn.Conv3d(in_dim, out_dim, kernel, stride)
        self.layernorm = nn.LayerNorm(feature_shape)
    
    def forward(self, x):
        x = self.transconv(x)
        x = self.layernorm(x)
        return nn.functional.leaky_relu(x)

class D(nn.Module):
    
    def __init__(self):
        super().__init__()
        # a different layer for each track
        self.conv0 = nn.ModuleList([
            DiscriminatorBlock(1, 16, kernel=(1, 1, 12), stride=(1, 1, 12), feature_shape=[16, 4, 16, 6])
            for _ in range(N_TRACKS)
        ])
        self.conv1 = nn.ModuleList([
            DiscriminatorBlock(16, 16, kernel=(1, 4, 1), stride=(1, 4, 1), feature_shape=[16, 4, 4, 6])
            for _ in range(N_TRACKS)
        ])
        # common layers from now on
        self.conv2 = DiscriminatorBlock(16 * 5, 64, kernel=(1, 1, 3), stride=(1, 1, 1), feature_shape=[64, 4, 4, 4])
        self.conv3 = DiscriminatorBlock(64, 64, kernel=(1, 1, 4), stride=(1, 1, 4), feature_shape=[64, 4, 4, 1])
        self.conv4 = DiscriminatorBlock(64, 128, kernel=(1, 4, 1), stride=(1, 4, 1), feature_shape=[128, 4, 1, 1])
        self.conv5 = DiscriminatorBlock(128, 128, kernel=(2, 1, 1), stride=(1, 1, 1), feature_shape=[128, 3, 1, 1])
        self.conv6 = DiscriminatorBlock(128, 256, kernel=(3, 1, 1), stride=(3, 1, 1), feature_shape=[256, 1, 1, 1])
        self.dense = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )
        


    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, N_TRACKS, N_MEASURES_FOR_SAMPLE, MEASURE_LENGTH, N_PITCHES)
        # a different layer for each track
        x_list = [conv(x[:, [i]]) for i, conv in enumerate(self.conv0)]
        x_list = [conv(x_) for x_, conv in zip(x_list, self.conv1)]
        # merge the branches' results concatenating them
        x = torch.cat(x_list, dim=1)
        x = self.conv2(x)
        x = self.conv3(x)     
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(batch_size, 256)
        x = self.dense(x)
        return x

