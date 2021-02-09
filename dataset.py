from pypianoroll import multitrack
import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
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

        # Used to load ALL the dataset
        for (path, name, file) in os.walk(music_folder):
            if(len(file) > 0):
                list_of_ids.append(path.split("/")[-1])

        # Use this if you want to load only specific tracks
        # list_of_files = os.listdir(os.path.join(self.dataset_path, "amg"))
        
        # for file in list_of_files:
        #     with open(os.path.join(self.dataset_path, "amg", file), 'r') as f:
        #         for id in f:
        #             list_of_ids.append(id.strip())



        # Load all the dataset in multitracks
        multitracks = []
        for i, ids in enumerate(list_of_ids):
            path = os.path.join(music_folder, from_id_to_path(ids))

            file_name = os.listdir(path)[0]
            file_path = os.path.join(path, file_name)
            if(file_path not in multitracks):
                multitracks.append(file_path)
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

        #if not data: print(available_measures, target_n_samples) # DEBUG
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
        batch = [torch.tensor(b, device=DEVICE) for b in batch]
        return torch.cat(batch, dim=0)


    d = PRollDataset("data", test = False)
    print(len(d.multitracks_paths))
    exit()
    # RICORDATI STO COMANDO find . -name '*.txt' | xargs wc -l

    counter = 0
    for elem in d:
        if(counter % 100 == 0):
            print(counter)
        counter += elem.shape[0]

    print(counter)
    exit()

    dl = DataLoader(d, batch_size = 32 // N_SAMPLES_PER_SONG, collate_fn = collate)

    for elem in dl:
        print(elem)
        
"""

   1475 ./id_list_favorite.txt
      0 ./id_list_progressive-rock.txt
    730 ./id_list_cover.txt
   4693 ./id_list_rock.txt
    768 ./id_list_acoustic.txt
   1836 ./id_list_party.txt
   1252 ./id_list_sexy.txt
   1579 ./id_list_70s.txt
   1624 ./id_list_favourites.txt
    871 ./id_list_melancholy.txt
    669 ./id_list_piano.txt
   1314 ./id_list_chillout.txt
    411 ./id_list_lounge.txt
    677 ./id_list_instrumental.txt
    488 ./id_list_electro.txt
    792 ./id_list_relax.txt
   1257 ./id_list_chill.txt
   3227 ./id_list_love.txt
   1537 ./id_list_soul.txt
   1443 ./id_list_favourite.txt
    289 ./id_list_rap.txt
    316 ./id_list_downtempo.txt
   1688 ./id_list_mellow.txt
   1061 ./id_list_sad.txt
    461 ./id_list_punk.txt
    213 ./id_list_experimental.txt
   1496 ./id_list_catchy.txt
    123 ./id_list_hardcore.txt
    539 ./id_list_techno.txt
   1768 ./id_list_electronic.txt
   1448 ./id_list_british.txt
    413 ./id_list_ambient.txt
   1322 ./id_list_cool.txt
   1024 ./id_list_loved.txt
   1118 ./id_list_rnb.txt
      0 ./id_list_classic-rock.txt
   2904 ./id_list_dance.txt
   3339 ./id_list_favorites.txt
   1140 ./id_list_60s.txt
   2507 ./id_list_oldies.txt
   2205 ./id_list_beautiful.txt
    673 ./id_list_jazz.txt
    681 ./id_list_house.txt
   1733 ./id_list_awesome.txt
   1231 ./id_list_indie.txt
    669 ./id_list_folk.txt
   5808 ./id_list_pop.txt
      0 ./id_list_heard-on-pandora.txt
   2076 ./id_list_alternative.txt
   1453 ./id_list_00s.txt
   1814 ./id_list_american.txt
    878 ./id_list_electronica.txt
    763 ./id_list_amazing.txt
    912 ./id_list_guitar.txt
    477 ./id_list_hip-hop.txt
   1668 ./id_list_classic.txt
    911 ./id_list_country.txt
   1230 ./id_list_happy.txt
    649 ./id_list_trance.txt
   1984 ./id_list_90s.txt
   2328 ./id_list_80s.txt
   1253 ./id_list_fun.txt
    305 ./id_list_psychedelic.txt
    585 ./id_list_metal.txt
    198 ./id_list_reggae.txt
   1140 ./id_list_soundtrack.txt
   1051 ./id_list_female.txt
    620 ./id_list_blues.txt
    742 ./id_list_funk.txt

"""
