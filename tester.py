
import torch
from generator import G
from configuration import *
import numpy as np
from pypianoroll import Multitrack, StandardTrack
import matplotlib.pyplot as plt
import sys 

import os


def generated_song_to_img(generated_song, write_midi = False):
    tempo_array = np.full((4 * 4 * MEASURE_LENGTH, 1), TEMPO)
    samples = generated_song.transpose(1, 0, 2, 3).reshape(N_TRACKS, -1, N_PITCHES)
    tracks = []
    for idx, (program, is_drum, track_name) in enumerate(
        zip(PROGRAMS, IS_DRUMS, TRACK_NAMES)
    ):
        pianoroll = np.pad(
            samples[idx] > 0.5,
            ((0, 0), (LOWEST_PITCH, 128 - LOWEST_PITCH - N_PITCHES))
        )
        tracks.append(
            StandardTrack(
                name=track_name,
                program=program,
                is_drum=is_drum,
                pianoroll=pianoroll
            )
        )
    m = Multitrack(
        tracks=tracks,
        tempo=tempo_array,
        resolution=BEAT_RESOLUTION
    )
    m.binarize()
    if(write_midi):
        m.write("tester.mid")

    axs = m.plot()
    plt.gcf().set_size_inches((16, 8))
    for ax in axs:
        for x in range(
            MEASURE_LENGTH,
            4 * MEASURE_LENGTH * N_MEASURES_FOR_SAMPLE,
            MEASURE_LENGTH
        ):
            if x % (MEASURE_LENGTH * 4) == 0:
                ax.axvline(x - 0.5, color='k')
            else:
                ax.axvline(x - 0.5, color='k', linestyle='-', linewidth=1)

    if(not write_midi):
        canvas = plt.gca().figure.canvas
        canvas.draw()
        data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        image = data.reshape(canvas.get_width_height()[::-1] + (3,))
        image = np.moveaxis(image, 2, 0)
        
        return image

    else:
        plt.show()
    



if(__name__ == "__main__"):

    checkpoints_path = CHECKPOINT_PATH.split("/")[:-1]
    if(len(checkpoints_path) > 1):
        checkpoints_path = os.path.join(*checkpoints_path)
    else:
        checkpoints_path = checkpoints_path[0]

    

    checkpoints = os.listdir(checkpoints_path) 

    checkpoint_idx = len(checkpoints)

    if(len(sys.argv) > 1):
        checkpoint_idx = sys.argv[1]

    print(checkpoint_idx)
    checkpoint = torch.load(checkpoints_path + "/checkpoint_" + str(checkpoint_idx) + ".pt", map_location="cpu")


    g = G()
    g.load_state_dict(checkpoint["generator"])
    g.eval()
    noise = torch.randn(4, LATENT_DIM)


    samples = g(noise).cpu().detach().numpy()
    generated_song_to_img(samples, write_midi=True)


