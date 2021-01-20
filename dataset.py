
from torch.utils.data import random_split, IterableDataset, RandomSampler, DataLoader
from torch.utils.data import random_split
import os
import numpy as np


class MyDataset(IterableDataset):

    def __init__(self, dataset_path):
        """
            dataset_path: path to the dataset file [str]
        """
        super().__init__()
        
        self.inputs = [np.load(f)["arr_0"] for f in os.listdir(dataset_path)[:5]]

    def __iter__(self):
        for i in RandomSampler(self.inputs):
            yield self.inputs[i]
    
    # def __item__(self, i):
    #     return 

    def __len__(self):
        return len(self.inputs)


a = MyDataset("dataset")
print(a[1])
# for elem in a:
#     print(elem.shape)


# train_length = int(len(dataset) * 0.75)
# test_length = int(len(dataset) * 0.10)
# evaluation_length = len(dataset) - train_length - test_length
# train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, (train_length, test_length, evaluation_length))

# train_loader = DataLoader(MyDataset(*zip(*train_data)), batch_size=BATCH_SIZE)
# dev_loader = DataLoader(MyDataset(*zip(*dev_data)), batch_size=BATCH_SIZE)
# test_loader = DataLoader(MyDataset(*zip(*test_data)), batch_size=BATCH_SIZE)

# for inputs, labels in loader:
#     pass

