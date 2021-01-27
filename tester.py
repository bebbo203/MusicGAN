from numpy.core.numeric import outer
from torch.serialization import load
from generator import G
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from tqdm.auto import tqdm
import numpy as np
import os
from pr_to_midi_converter import piano_roll_to_pretty_midi


def multi_track_padder(instruments):
    ret = []
    for instrument in instruments:
        pad_array = np.zeros((1, instrument.shape[1], 128))
        pad_array[:, :, 29: 29 + instrument.shape[2]] += instrument
        ret.append(pad_array[0].T)
    return np.stack(ret, axis=0)


NOISE_SIZE = 100
# Load a checkpoint
checkpoint = torch.load("checkpoints/checkpoint_16.pt")
g = G(100, 69*4)
g.load_state_dict(checkpoint["generator"])

# Generate a song from normal noise
noise = torch.randn((1, 500, NOISE_SIZE))
generated_song = g(noise).detach().numpy()
generated_song *= 127
generated_song = np.rint(generated_song).astype(np.uint8)


# Pad the instruments to the right length (128)
instruments = np.array_split(generated_song, 4, axis=2)
multi_track = multi_track_padder(instruments)
midi = piano_roll_to_pretty_midi(multi_track, sf=100)
midi.write("tester.mid")


