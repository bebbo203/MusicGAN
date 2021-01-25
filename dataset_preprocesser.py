from dataset_extractor import DATASET_PATH
import os
import numpy as np
from tqdm import tqdm


MAX_EXTENSION = 68



def pad_song(multi_piano_roll, padding=128):
    res = np.zeros((4, padding, multi_piano_roll.shape[2]))
    from_idx = (padding - multi_piano_roll.shape[1]) // 2
    res[:, from_idx:from_idx+multi_piano_roll.shape[1], :] += multi_piano_roll
    return res


def preprocess(track):
       
        # Trim notes 
        def get_song_extension(multi_piano_roll):
            song_ext = 0
            song_min_note = 128
            song_max_note = 0
            for instrument in multi_piano_roll:
                notes_in_track_idx = np.argwhere(instrument != 0)[:, 0]
                _min_note = np.min(notes_in_track_idx)
                if(_min_note < song_min_note):
                    song_min_note = _min_note
                _max_note = np.max(notes_in_track_idx)
                if(_max_note > song_max_note):
                    song_max_note = _max_note
                extension = _max_note - _min_note
                if(extension > song_ext):
                    song_ext = extension
            
            return song_min_note, song_max_note

       
        lowest_note, highest_note = get_song_extension(track)

        if(highest_note - lowest_note > MAX_EXTENSION):
            return None

        track = track[:, lowest_note:highest_note+1, :]
        track = pad_song(track, MAX_EXTENSION+1)
        

        # Remove silence 
        initial_silence = track.shape[-1]
        ending_silence = 0
        for instrument in track:
            track_initial_silence = np.min(np.argwhere(instrument != 0)[:,1])
            track_ending_silence = np.max(np.argwhere(instrument != 0)[:,1])

            if(track_initial_silence < initial_silence):
                initial_silence = track_initial_silence
            if(track_ending_silence > ending_silence):
                ending_silence = track_ending_silence

        # shape: (instruments, notes, time)
        track_without_silences = track[:, :, initial_silence : ending_silence]
        # shape: (time, instruments * notes)
        track_without_silences = track_without_silences.reshape((-1, track_without_silences.shape[-1])).T

        return track_without_silences


DATASET_PATH = "dataset"
files_path = []
test_dataset = False
for root, dirs, files in os.walk(DATASET_PATH, topdown=False):
    for name in files:
        files_path.append(os.path.join(root, name))
    if(test_dataset and len(files_path) >= 10):
        break


for i, file_path in tqdm(enumerate(files_path), total = len(files_path)):
    _track = np.load(file_path)["arr_0"]
    track = preprocess(_track)

    if(track is not None): 
        track = track.astype(np.uint8)
        #np.savez_compressed("dataset_preprocessed_test/track_"+str(i)+".npz", track)
        np.save("dataset_preprocessed_uncompressed/track_"+str(i)+".npy", track)
        

