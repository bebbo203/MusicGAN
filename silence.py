from converter import piano_roll_to_pretty_midi
import numpy as np

total_dim = 0
total_silence_removed = 0
for i in range(5000):
    multi_piano_roll = np.load("dataset/track_"+str(i)+".npz")["arr_0"]

    initial_silence = multi_piano_roll.shape[-1]
    ending_silence = 0
    for track in multi_piano_roll:
        track_initial_silence = np.min(np.argwhere(track != 0)[:,1])
        track_ending_silence = np.max(np.argwhere(track != 0)[:,1])

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



