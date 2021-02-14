import torch
from generator import G
from configuration import *
import numpy as np
from pypianoroll import Multitrack, StandardTrack
import matplotlib.pyplot as plt
import sys 
import os
from PIL import Image



def generated_song_to_img(generated_song, write_midi=False):
    """
    Generate or plot the pianoroll image
    
        Parameters:
            generated_song (torch.tensor): the output of a Generator network
            writi_midi (bool): if true a .mid file will be saved and the plot showed, 
                               if false return the image of the plot
        
        Returns:
            image (numpy.ndarray): if write_midi=False return the plot image
    """

    # Array used by the Multitrack class to detect the tempo of the song. In our
    # case the song is played always with the same speed. 
    tempo_array = np.full((4 * 4 * MEASURE_LENGTH, 1), TEMPO)
    # Reshape the song to the format used by the Track object
    samples = generated_song.transpose(1, 0, 2, 3).reshape(N_TRACKS, -1, N_PITCHES)
    tracks = []
    for idx, (program, is_drum, track_name) in enumerate(zip(PROGRAMS, IS_DRUMS, TRACK_NAMES)):
        # Tracks object work on 128 pitches while the network on N_PITCHES only
        # Binarize and then pad to meet the required dimensions
        pianoroll = np.pad(samples[idx] > 0.5, ((0, 0), (LOWEST_PITCH, 128 - LOWEST_PITCH - N_PITCHES)))
        # Generates the track object
        st = StandardTrack(name=track_name, program=program, is_drum=is_drum, pianoroll=pianoroll)
        tracks.append(st)

    # Fill a multitrack object with the tracks generated in the step before
    m = Multitrack(tracks=tracks, tempo=tempo_array, resolution=BEAT_RESOLUTION)
    m.binarize()
    
    if(write_midi):
        m.write(GENERATED_PATH)

    # Plot the pianoroll
    axs = m.plot()
    plt.gcf().set_size_inches((16, 8))
   
    # If you don't want a midi then return an img of the plot (used to log)
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

    # Load the path with the checkpoints of the training
    checkpoints_path = CHECKPOINT_PATH.split("/")[:-1]
    if(len(checkpoints_path) > 1):
        checkpoints_path = os.path.join(*checkpoints_path)
    else:
        checkpoints_path = checkpoints_path[0]
    checkpoints = os.listdir(checkpoints_path)  
    checkpoint_idx = len(checkpoints)
    
    # If an integer is present as a command argument, take that epoch
    # otherwise take the last epoch
    if(len(sys.argv) > 1):
        checkpoint_idx = sys.argv[1]

    # Print and ack of the epoch taken
    print(checkpoint_idx)
    checkpoint = torch.load(checkpoints_path + "/checkpoint_" + str(checkpoint_idx) + ".pt", map_location="cpu")


    # Instantiate the generator model and set it to the eval mode
    g = G().to(DEVICE)
    g.load_state_dict(checkpoint["generator"])
    g.eval()
    
    # Define a normal random vector as input noise
    noise = torch.randn(4, LATENT_DIM)
    # Generate and store the MIDI file
    with torch.no_grad():
        samples = g(noise).cpu().numpy()
        generated_song_to_img(samples, write_midi=True)

