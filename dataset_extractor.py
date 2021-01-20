import pypianoroll
import numpy as np
import os
from tqdm.auto import tqdm


files_path = []
DATASET_PATH = "./raw_dataset"
test_dataset = False
for root, dirs, files in os.walk(DATASET_PATH, topdown=False):
    for name in files:
        files_path.append(os.path.join(root, name))
    if(test_dataset and len(files_path) >= 10):
        break



#[14601, 13524, 14839, 17775, 5271]
instruments_order = [
    # Piano / Organ
    [0, 1, 4, 2, 6, 7, 18],
    # Ensemble
    [48, 52, 49, 50, 51],
    # Guitar
    [25, 27, 30, 29, 26, 24, 28, 31],
    # Bass
    [33, 35, 32, 38, 39, 34]
]


track_number = 0
for file in tqdm(files_path):
    multitrack = pypianoroll.load(file)
    # to be filled with the pianoroll relatives to the instrument
    tracks_piano_rolls = []
    # dict in the form {"channel": Track}
    track_with_instrument = {}
    for track in multitrack.tracks:
        if(track.is_drum == False):
            track_with_instrument.update({str(track.program): track.pianoroll})
    
    for class_code in instruments_order:
        for instrument in class_code:
            if(str(instrument) in track_with_instrument.keys()):
                tracks_piano_rolls.append(np.transpose(track_with_instrument[str(instrument)]))
                break

    # the files are saved in a compressed archive
    # to retrieve: arr = np.load(PATH)["arr_0"]
    if(len(tracks_piano_rolls) == len(instruments_order)):
        output_song = np.stack(tracks_piano_rolls)
        np.savez_compressed("dataset/track_"+str(track_number)+".npz", output_song)
        track_number += 1