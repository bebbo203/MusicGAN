from numpy.core import multiarray
import pypianoroll
import matplotlib.pyplot as plt
import numpy as np
import os
from converter import *
from tqdm import tqdm


for i, f in enumerate(os.listdir("dataset_preprocessed")):
   l = np.load(os.path.join("./dataset_preprocessed", f))
   print(l.shape)

exit()




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


def pad_song(multi_piano_roll):
    res = np.zeros((4, 128, multi_piano_roll.shape[2]))
    from_idx = (128 - multi_piano_roll.shape[1]) // 2
    res[:, from_idx:from_idx+multi_piano_roll.shape[1], :] += multi_piano_roll
    return res




multi_piano_roll = np.load("dataset/track_10.npz")["arr_0"]
midi = piano_roll_to_pretty_midi(multi_piano_roll)
midi.write("before_trim.mid")


m, M = get_song_extension(multi_piano_roll)


multi_piano_roll = multi_piano_roll[:, m:M+1, :]

multi_piano_roll = pad_song(multi_piano_roll)
midi = piano_roll_to_pretty_midi(multi_piano_roll)
midi.write("after_trim.mid")





exit()
"""
{'0': 17535, '48': 9090, '25': 8256, '33': 7863, '27': 6159, '30': 5981, '52': 5404, '29': 5124, '49': 5068, 
'35': 4562, '26': 4052, '24': 4023, '50': 3925, '81': 3470, '28': 3341, '1': 3204, '53': 3078, '32': 2766, 
'73': 2659, '38': 2559, '65': 2488, '4': 2470, '61': 2429, '119': 2294, '18': 2054, '5': 1991, '56': 1865, 
'62': 1814, '89': 1777, '54': 1743, '39': 1687, '87': 1643, '66': 1639, '80': 1588, '11': 1553, '57': 1462, 
'17': 1436, '34': 1434, '2': 1330, '88': 1323, '16': 1252, '90': 1246, '51': 1143, '60': 1080, '95': 1021, 
'75': 1004, '68': 971, '82': 964, '71': 963, '100': 924, '6': 904, '46': 895, '7': 886, '40': 879, '91': 843, 
'45': 836, '122': 828, '99': 790, '63': 772, '22': 740, '47': 713, '3': 680, '94': 636, '55': 584, '64': 573, 
'85': 570, '44': 570, '118': 560, '36': 552, '42': 551, '21': 545, '78': 511, '72': 498, '12': 482, '127': 467, 
'67': 463, '79': 447, '120': 446, '14': 443, '31': 433, '9': 433, '84': 416, '102': 402, '10': 392, '19': 378, 
'37': 378, '70': 366, '8': 353, '74': 353, '59': 347, '58': 343, '93': 338, '41': 336, '96': 330, '125': 322, 
'43': 316, '105': 304, '117': 299, '98': 282, '77': 280, '92': 280, '126': 275, '104': 266, '124': 257, '103': 249, 
'83': 244, '23': 241, '116': 226, '115': 213, '110': 212, '69': 205, '114': 200, '106': 198, '20': 182, '13': 167, 
'101': 161, '108': 160, '113': 153, '112': 152, '76': 151, '107': 141, '86': 139, '123': 127, '97': 119, '15': 114, 
'121': 104, '109': 83, '111': 58}
"""





file_path = []
DATASET_PATH = "./raw_dataset"
for root, dirs, files in os.walk(DATASET_PATH, topdown=False):
    for name in files:
        file_path.append(os.path.join(root, name))
     

instruments = {}

# The track numbers are minus one w.r.t the one on wikipedia
for song_path in file_path:
    multitrack = pypianoroll.load(song_path)
    for track in multitrack.tracks:
        
        if(track.program == 0):
            print(track)

        continue
        if(track.is_drum == False):
            if(str(track.program) in instruments):
                instruments[str(track.program)] += 1
            else:
                instruments.update({str(track.program): 1})

print(instruments)
        

