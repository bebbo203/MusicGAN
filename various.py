#from converter import piano_roll_to_pretty_midi
import numpy as np
from tqdm import tqdm

# How many songs have the i-th extension
l = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 4, 0, 3, 1, 2, 2, 4, 8, 16, 11, 20, 20, 53, 14, 89, 33, 65, 112, 25, 189, 73, 159, 216, 116, 386, 58, 335, 176, 203, 353, 88, 449, 172, 199, 295, 126, 420, 105, 287, 193, 187, 295, 54, 369, 85, 176, 104, 48, 131, 13, 47, 28, 19, 27, 7, 27, 0, 22, 5, 2, 11, 1, 7, 3, 4, 7, 1, 2, 0, 0, 0, 3, 5, 7, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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

for i in tqdm(range(6789)):
    multi_piano_roll = np.load("dataset/track_"+str(i)+".npz")["arr_0"]
    song_ext = 0
    min_note = 500
    max_note = 0
    for track in multi_piano_roll:
        argw = np.argwhere(track != 0)[:, 0]
        _min_note = np.min(argw)
        _max_note = np.max(argw)

        if(_min_note < min_note):
            min_note = _min_note
        if(_max_note > max_note):
            max_note = _max_note

    extension = max_note - min_note
    l[extension] += 1
        
            
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



