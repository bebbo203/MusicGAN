import pypianoroll
import os

from torch.utils.data import dataset

def from_id_to_path(ids):
    return os.path.join(ids[2], ids[3], ids[4], ids)

dataset_path = "data"

music_folder = os.path.join(dataset_path, "lpd_5", "lpd_5_cleansed")
list_of_files = os.listdir(os.path.join(dataset_path, "lastfm"))
list_of_ids = []
    
for file in list_of_files:
    if(file in ["id_list_dance.txt"]):
        with open(os.path.join(dataset_path, "lastfm", file), 'r') as f:
            for id in f:
                list_of_ids.append(id.strip())


# Load all the dataset in multitracks
multitracks = []
for i, ids in enumerate(list_of_ids):
    path = os.path.join(music_folder, from_id_to_path(ids))
    file_name = os.listdir(path)[0]
    multitracks.append(os.path.join(path, file_name))
    

song = pypianoroll.load(multitracks[3])
song.write("tester.mid")
