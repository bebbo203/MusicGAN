import torch
import numpy as np
from torch.utils.data import random_split, IterableDataset, RandomSampler, DataLoader
import os
from tqdm import tqdm

class PRollDataset(IterableDataset):
    def __init__(self, dataset_path, device = "cuda", test = False):
        """
            dataset_path: path to the dataset file (str)
            device: torch device (str) ["cuda" | "cpu"]
            test: boolean value to fetch less data
        """
        super().__init__()
        
        self.device = torch.device(device)
        self.dataset_path = dataset_path
        if test:
            self.file_paths = os.listdir(dataset_path)[:200]
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

    def __getitem__(self, index):
        track = np.load(os.path.join(self.dataset_path, self.file_paths[index])).astype(np.float32)
        track = torch.tensor(track).to(self.device)
        track = track + torch.randn_like(track)
        track = torch.abs(track / 127.0) 
        return track
    
