import torch
import numpy as np
from torch.utils.data import random_split, IterableDataset, RandomSampler, DataLoader
import os
from tqdm import tqdm

class PRollDataset(IterableDataset):
    def __init__(self, dataset_path, device = "cuda", test = False):
        """
            dataset_path: path to the dataset file [str]
        """
        super().__init__()
        
        self.device = torch.device(device)
        self.dataset_path = dataset_path
        if test:
            self.file_paths = os.listdir(dataset_path)[:2000]
        else:
            self.file_paths = os.listdir(dataset_path)

    
    def __len__(self):
        return len(self.file_paths)

    # Remember that with the noise some values are negative
    def __iter__(self):
        for i in RandomSampler(self.file_paths):
            track = np.load(os.path.join(self.dataset_path, self.file_paths[i])).astype(np.float32)
            track = torch.tensor(track).to(self.device)
            track = track + torch.randn_like(track)
            track = torch.abs(track / 127.0) 
            yield track
    
d = PRollDataset("dataset_preprocessed_test", test=True)


for elem in tqdm(d):
    continue