import torch.nn as nn
import torch
from torch.nn.modules import dropout
from dataset import PRollDataset

class G(nn.Module):
    def __init__(self, noise_size, output_size):
        super().__init__()
        
        ngf = 64
        final_channels_number = 1  # TODO: are 4 channels, one per instrument, better than one big channel?



        self.conv1 = nn.Sequential(     
            nn.ConvTranspose2d(in_channels = noise_size,
                               out_channels = ngf * 8,
                               kernel_size = 6,
                               stride = 1, 
                               padding = 0,
                               bias = False),      
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True)
        )

        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = ngf * 8,
                               out_channels = ngf * 4,
                               kernel_size = 7,
                               stride = 7,
                               padding = (2, 4),
                               bias = False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )

        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = ngf * 4,
                               out_channels = ngf * 2,
                               kernel_size = 5,
                               stride = (3, 3),
                               padding = (1, 3),
                               bias = False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = ngf * 2,
                               out_channels = ngf,
                               kernel_size = 5,
                               stride = (3, 1), 
                               padding = (5, 4),
                               bias = False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )
        
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = ngf,
                               out_channels = final_channels_number,
                               kernel_size = 5,
                               stride = (3, 3),
                               padding = (2, 4),
                               bias = False),
            nn.Sigmoid()
        )
       


    def forward(self, noise):
        o = self.conv1(noise)
        o = self.conv2(o)
        o = self.conv3(o)
        o = self.conv4(o)
        o = self.conv5(o)

        return o



if(__name__ == "__main__"):

    d = PRollDataset("dataset_preprocessed_reduced", test=True)
    g = G(100, 1)

    elem = d[0]
    noise = torch.randn(1, 100, 1, 1)
    g(noise)

