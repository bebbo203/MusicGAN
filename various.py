#from converter import piano_roll_to_pretty_midi
import numpy as np
from tqdm import tqdm

# How many songs have the i-th extension
l = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 7, 75, 18, 34, 38, 41, 94, 29, 151, 71, 152, 141, 69, 352, 71, 279, 216, 183, 342, 64, 404, 137, 214, 188, 88, 330, 61, 237, 154, 161, 245, 59, 291, 76, 149, 145, 63, 240, 42, 134, 64, 116, 121, 12, 133, 47, 50, 73, 21, 65, 12, 27, 24, 17, 21, 3, 32, 8, 14, 7, 1, 18, 6, 5, 4, 5, 4, 0, 6, 0, 1, 2, 1, 3, 1, 3, 6, 0, 3, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
m = 0

for i in range(len(l)):
    how_much_off = 0
    for j in range(i, len(l)):
        how_much_off += l[j]
    print(f"{i}: {how_much_off} ({how_much_off / 6789 * 100:.2f}%)")










exit()
# total_dim = 0
# total_silence_removed = 0


# Song extension

l = [0] * 128
max_extension = 0
min_note = 500
max_note = 0
for i in tqdm(range(6789)):
    multi_piano_roll = np.load("dataset/track_"+str(i)+".npz")["arr_0"]
    song_ext = 0
    for track in multi_piano_roll:
        argw = np.argwhere(track != 0)[:, 0]
        _min_note = np.min(argw)
        _max_note = np.max(argw)
        extension = _max_note - _min_note
        if(extension > song_ext):
            song_ext = extension
    l[song_ext] += 1
        
            
print(l)




exit()
# Silence removal
for i in range(5000):
    multi_piano_roll = np.load("dataset/track_"+str(i)+".npz")["arr_0"]

    initial_silence = multi_piano_roll.shape[-1]
    ending_silence = 0
    for track in multi_piano_roll:
        track_initial_silence = np.min(np.argwhere(track != 0)[:,1])
        track_ending_silence = np.max(np.argwhere(track !=
         0)[:,1])

        if(track_initial_silence < initial_silence):
            initial_silence = track_initial_silence
        if(track_ending_silence > ending_silence):
            ending_silence = track_ending_silence
    
    total_dim += multi_piano_roll[0].shape[-1]
    total_silence_removed += initial_silence + (multi_piano_roll[0].shape[-1] - ending_silence)


    

print(total_dim)
print(total_silence_removed)
exit()





multi_piano_roll = multi_piano_roll[:, :, initial_silence: ending_silence]


midi = piano_roll_to_pretty_midi(multi_piano_roll)
midi.write("prova_cut.mid")



