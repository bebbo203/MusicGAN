from typing import MutableMapping
import torch
import numpy as np
from torch.utils.data import random_split, IterableDataset, RandomSampler, DataLoader
import os


# L'idea è quella di iniettare nel dataset un po' di noise
# Ricordiamo che la LSTM e la dense non hanno limiti nell'output:
# significa che il gradiente fluisce e a fine training ci basta fare una
# round per ottenere la canzone vera

class PRollDataset(IterableDataset):

    def __init__(self, dataset_path, device = "cuda", test = False):
        """
            dataset_path: path to the dataset file [str]
        """
        super().__init__()
        
        self.device = torch.device(device)
        
        if test:
            raw_inputs = [np.load(os.path.join(dataset_path, f))["arr_0"] for f in os.listdir(dataset_path)[:100]]
        else:
            raw_inputs = [np.load(os.path.join(dataset_path, f))["arr_0"] for f in os.listdir(dataset_path)]

        self.inputs = self.preprocess(raw_inputs)

    def preprocess(self, raw_inputs):
        # Remove silence 
        preprocessed_inputs = []
        for multitrack in raw_inputs:
            initial_silence = multitrack.shape[-1]
            ending_silence = 0
            for track in multitrack:
                track_initial_silence = np.min(np.argwhere(track != 0)[:,1])
                track_ending_silence = np.max(np.argwhere(track != 0)[:,1])

                if(track_initial_silence < initial_silence):
                    initial_silence = track_initial_silence
                if(track_ending_silence > ending_silence):
                    ending_silence = track_ending_silence

            # (instruments, notes, time)
            multitrack_without_silences = torch.FloatTensor(multitrack[:, :, initial_silence : ending_silence]).to(self.device)
            # (time, instruments * notes)
            multitrack_without_silences = multitrack_without_silences.view((-1, multitrack_without_silences.shape[-1])).transpose(0,1)

            preprocessed_inputs.append(multitrack_without_silences)

        return preprocessed_inputs
        # TODO: togli le ottave di troppo

    # Remember that with the noise some values are negative
    def __iter__(self):
        for i in RandomSampler(self.inputs):
            yield self.inputs[i] + torch.randn_like(self.inputs[i])
    
    def __getitem__(self, i):
        return self.inputs[i] 

    def __len__(self):
        return len(self.inputs)





# train_length = int(len(dataset) * 0.75)
# test_length = int(len(dataset) * 0.10)
# evaluation_length = len(dataset) - train_length - test_length
# train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, (train_length, test_length, evaluation_length))

# train_loader = DataLoader(MyDataset(*zip(*train_data)), batch_size=BATCH_SIZE)
# dev_loader = DataLoader(MyDataset(*zip(*dev_data)), batch_size=BATCH_SIZE)
# test_loader = DataLoader(MyDataset(*zip(*test_data)), batch_size=BATCH_SIZE)

# for inputs, labels in loader:
#     pass

