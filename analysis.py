#from converter import piano_roll_to_pretty_midi
import numpy as np
from tqdm import tqdm




# Instruments count

# d = {'0': 17535, '48': 9090, '25': 8256, '33': 7863, '27': 6159, '30': 5981, '52': 5404, '29': 5124, '49': 5068, 
# '35': 4562, '26': 4052, '24': 4023, '50': 3925, '81': 3470, '28': 3341, '1': 3204, '53': 3078, '32': 2766, 
# '73': 2659, '38': 2559, '65': 2488, '4': 2470, '61': 2429, '119': 2294, '18': 2054, '5': 1991, '56': 1865, 
# '62': 1814, '89': 1777, '54': 1743, '39': 1687, '87': 1643, '66': 1639, '80': 1588, '11': 1553, '57': 1462, 
# '17': 1436, '34': 1434, '2': 1330, '88': 1323, '16': 1252, '90': 1246, '51': 1143, '60': 1080, '95': 1021, 
# '75': 1004, '68': 971, '82': 964, '71': 963, '100': 924, '6': 904, '46': 895, '7': 886, '40': 879, '91': 843, 
# '45': 836, '122': 828, '99': 790, '63': 772, '22': 740, '47': 713, '3': 680, '94': 636, '55': 584, '64': 573, 
# '85': 570, '44': 570, '118': 560, '36': 552, '42': 551, '21': 545, '78': 511, '72': 498, '12': 482, '127': 467, 
# '67': 463, '79': 447, '120': 446, '14': 443, '31': 433, '9': 433, '84': 416, '102': 402, '10': 392, '19': 378, 
# '37': 378, '70': 366, '8': 353, '74': 353, '59': 347, '58': 343, '93': 338, '41': 336, '96': 330, '125': 322, 
# '43': 316, '105': 304, '117': 299, '98': 282, '77': 280, '92': 280, '126': 275, '104': 266, '124': 257, '103': 249, 
# '83': 244, '23': 241, '116': 226, '115': 213, '110': 212, '69': 205, '114': 200, '106': 198, '20': 182, '13': 167, 
# '101': 161, '108': 160, '113': 153, '112': 152, '76': 151, '107': 141, '86': 139, '123': 127, '97': 119, '15': 114, 
# '121': 104, '109': 83, '111': 58}



# categories = [0] * 16

# for _elem in d:
#     elem = int(_elem)
#     if(elem <= 7):
#         categories[0] += d[_elem]
#     elif(elem >= 8 and elem <=15):
#         categories[1] += d[_elem]
#     elif(elem >=16 and elem <= 23):
#         categories[2] += d[_elem]
#     elif(elem >=24 and elem <= 31):
#         categories[3] += d[_elem]
#     elif(elem >=32 and elem <= 39):
#         categories[4] += d[_elem]
#     elif(elem >=40 and elem <= 47):
#         categories[5] += d[_elem]
#     elif(elem >=48 and elem <= 55):
#         categories[6] += d[_elem]
#     elif(elem >=56 and elem <= 63):
#         categories[7] += d[_elem]
#     elif(elem >=64 and elem <= 71):
#         categories[8] += d[_elem]
#     elif(elem >=72 and elem <= 79):
#         categories[9] += d[_elem]
#     elif(elem >=80 and elem <= 87):
#         categories[10] += d[_elem]
#     elif(elem >=88 and elem <= 95):
#         categories[11] += d[_elem]
#     elif(elem >=96 and elem <= 103):
#         categories[12] += d[_elem]
#     elif(elem >=104 and elem <= 111):
#         categories[13] += d[_elem]
#     elif(elem >=112 and elem <= 119):
#         categories[14] += d[_elem]
#     elif(elem >=120 and elem <= 127):
#         categories[15] += d[_elem]

# print(categories)


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


# Song extension counter

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
# Silence removal counter
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



