
from operator import mul
from cnn_generator import G
import torch
import numpy as np
from PIL import Image
from pr_to_midi_converter import piano_roll_to_pretty_midi





def multi_track_padder(instruments):
    ret = []
    for instrument in instruments:
        pad_array = np.zeros((1, instrument.shape[1], 128))
        pad_array[:, :, 29: 29 + instrument.shape[2]] += instrument
        ret.append(pad_array[0].T)
    return np.stack(ret, axis=0)


NOISE_SIZE = 3
# Load a checkpoint
checkpoint = torch.load("checkpoints/checkpoint_9.pt", map_location="cpu")
g = G(NOISE_SIZE, 69*4)
g.load_state_dict(checkpoint["generator"])

# Generate a song from normal noise
noise = torch.randn((1, NOISE_SIZE, 1, 1))
generated_song = g(noise).detach().numpy()[0]
generated_song *= 127
generated_song = np.rint(generated_song).astype(np.uint8)

# Pad the instruments to the right length (128)
instruments = np.array_split(generated_song, 4, axis=2)
multi_track = multi_track_padder(instruments)
midi = piano_roll_to_pretty_midi(multi_track, sf=100)
midi.write("tester.mid")

# Create a colored background for the instruments
black_background = np.zeros((3, 1000, 276)).astype(np.uint8)
black_background[0, :, :69] = 100
black_background[1, :, 69:138] = 100
black_background[2, :, 138:207] = 100
black_background[1, :, 207:376] = 100
black_background[2, :, 207:276] = 100
black_background = np.rollaxis(black_background, 0, 3)





for i,elem in enumerate(instruments):
    elem = elem[0]
    black_background[:, 69*i:69*(i+1) , 0] += elem.astype(np.uint8)
    black_background[:, 69*i:69*(i+1) , 1] += elem.astype(np.uint8)
    black_background[:, 69*i:69*(i+1) , 2] += elem.astype(np.uint8)


    
img = Image.fromarray(black_background)
img.save("tester.png")

