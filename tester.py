
from generator import G
import torch
import numpy as np
from PIL import Image
from pr_to_midi_converter import piano_roll_to_pretty_midi


def padder(batch):
    max_dim = 0 
    for elem in batch:
        if(elem.shape[0] > max_dim):
            max_dim = elem.shape[0]
    
    for i in range(len(batch)):
        batch[i] = F.pad(batch[i], pad=(0, 0, 0, max_dim - batch[i].shape[0]))
        #batch[i] = batch[i][:500, :]

    return torch.stack(batch, dim=0)


def multi_track_padder(instruments):
    ret = []
    for instrument in instruments:
        pad_array = np.zeros((1, instrument.shape[1], 128))
        pad_array[:, :, 29: 29 + instrument.shape[2]] += instrument
        ret.append(pad_array[0].T)
    return np.stack(ret, axis=0)


NOISE_SIZE = 3
# Load a checkpoint
checkpoint = torch.load("checkpoints/checkpoint_56.pt", map_location="cpu")
g = G(NOISE_SIZE, 69*4)
g.load_state_dict(checkpoint["generator"])

# Generate a song from normal noise
noise = torch.randn((1, 1000, NOISE_SIZE))
generated_song = g(noise).detach().numpy()
generated_song *= 127
generated_song = np.rint(generated_song).astype(np.uint8)


# Pad the instruments to the right length (128)
instruments = np.array_split(generated_song, 4, axis=2)
multi_track = multi_track_padder(instruments)
midi = piano_roll_to_pretty_midi(multi_track, sf=100)
midi.write("tester.mid")

# Generate an image for the pianoroll
img = Image.fromarray(generated_song[0], "L")
img.save("tester.png")

