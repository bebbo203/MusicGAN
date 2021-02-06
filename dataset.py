from pypianoroll import multitrack
import torch
import numpy as np
from torch.utils.data import IterableDataset
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm
import os
import pypianoroll
from configuration import *


class PRollDataset(IterableDataset):
    def __init__(self, dataset_path, device="cuda", test = False):
        self.dataset_path = dataset_path
        self.device = device
        self.test = test

        self.beat_resolution = BEAT_RESOLUTION
        self.measure_length = MEASURE_LENGTH
        self.lowest_pitch = LOWEST_PITCH
        self.n_pitches = N_PITCHES
        self.n_measures_for_sample = N_MEASURES_FOR_SAMPLE
        self.n_samples_per_song = N_SAMPLES_PER_SONG
        self.latent_dim = LATENT_DIM
        self.n_tracks = N_TRACKS


        self.multitracks_paths = self.load_multitracks()

    
    def load_multitracks(self):
        def from_id_to_path(ids):
            return os.path.join(ids[2], ids[3], ids[4], ids)

        multitracks = []
        list_of_ids = []
        
        music_folder = os.path.join(self.dataset_path, "lpd_5", "lpd_5_cleansed")
        list_of_files = os.listdir(os.path.join(self.dataset_path, "amg"))
        for file in list_of_files:
            with open(os.path.join(self.dataset_path, "amg", file), 'r') as f:
                for id in f:
                    list_of_ids.append(id.strip())

        # Load all the dataset in multitracks
        multitracks = []
        for i, ids in enumerate(list_of_ids):
            path = os.path.join(music_folder, from_id_to_path(ids))
            file_name = os.listdir(path)[0]
            multitracks.append(os.path.join(path, file_name))
            #multitracks.append(pypianoroll.load(os.path.join(path, file_name)))
            if(self.test and i > 100):
                break

        return multitracks

    def take_samples_from_multitrack(self, multitrack):
        # Make the pianoroll of boolean type
        multitrack.binarize()
        # Set the resolution of a beat (in this case the min length of a note is 1/4)
        multitrack.set_resolution(self.beat_resolution)
        # Get the complete pianoroll (shape: n_tracks x n_timesteps x n_pitches)
        pr = multitrack.stack() > 0
        # Pick only the center pitches
        pr = pr[:, :, self.lowest_pitch:self.lowest_pitch + self.n_pitches]

        n_total_measures = multitrack.get_max_length() // self.measure_length
        available_measures = n_total_measures - self.n_measures_for_sample
        target_n_samples = min(n_total_measures // self.n_measures_for_sample, self.n_samples_per_song)

        data = []
        for idx in np.random.choice(available_measures, target_n_samples, False):
            start = idx * self.measure_length
            end = (idx + self.n_measures_for_sample) * self.measure_length
            # At least one instrument in the sample must have at least 10 notes
            if((pr[:, start:end].sum(axis=(1, 2)) > 10).any()):
                data.append(pr[:, start:end])

        #if not data: print(available_measures, target_n_samples)
        return np.stack(data) if data else None

    def __iter__(self):
        for i in RandomSampler(self.multitracks_paths):
            multitrack = pypianoroll.load(self.multitracks_paths[i])
            data = self.take_samples_from_multitrack(multitrack)
            
            if data is not None: yield data

    def __getitem__(self, index):
        path = self.multitracks_paths[index]
        multitrack = pypianoroll.load(path)
        return self.take_samples_from_multitrack(multitrack)

    def __len__(self):
        return len(self.multitracks_paths) * self.n_samples_per_song


if(__name__ == "__main__"):

    def collate(batch):
        batch = [torch.tensor(b, device="cuda") for b in batch]
        return torch.cat(batch, dim=0)


    d = PRollDataset("data", test = True)

    dl = torch.utils.data.DataLoader(d, batch_size = 32 // N_SAMPLES_PER_SONG, collate_fn = collate)

    for elem in dl:
        print(elem)
        
