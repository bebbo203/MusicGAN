import pretty_midi
import numpy as np
import pypianoroll



# Thanks to https://github.com/craffel/pretty-midi/pull/129/commits/8737653678b6835b945769ca60a6ddef226f1143
# remember to transpose the pianoroll


def piano_roll_to_pretty_midi(multi_piano_roll, sf=50):
    instruments = []
    program_nums = [0, 48, 25, 33, 1]
    pm = pretty_midi.PrettyMIDI()
    
    for piano_roll in multi_piano_roll:
        instruments.append(_piano_roll_to_pretty_midi(piano_roll, sf, program_num = program_nums[len(instruments)]))

    for instrument in instruments:
        pm.instruments.append(instrument)

    return pm


def _piano_roll_to_pretty_midi(piano_roll, sf=100, program_num=1):
    """Convert piano roll in the form (n_instruments, pitch, time)
       to a pretty_midi object"""

    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program_num)

    piano_roll = np.hstack((np.zeros((notes, 1)),
                            piano_roll,
                            np.zeros((notes, 1))))

    velocity_changes = np.nonzero(np.diff(piano_roll).T)
    current_velocities = np.zeros(notes,dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        velocity = piano_roll[note, time + 1]
        time = time / sf
        if velocity > 0:
            if current_velocities[note] == 0:
                note_on_time[note] = time
                current_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                    velocity=current_velocities[note],
                    pitch=note,
                    start=note_on_time[note],
                    end=time)

            instrument.notes.append(pm_note)
            current_velocities[note] = 0
    return instrument


"""
pm = pretty_midi.PrettyMIDI()

p = pypianoroll.load("prova.npy")
i = []

for track in p.tracks:
    i.append(track.pianoroll)

multi_piano_roll = np.stack(i)

multi_piano_roll = np.load("dataset/track_341.npz")["arr_0"]
multi_piano_roll = multi_piano_roll[:, :, 700:1200]
midi = piano_roll_to_pretty_midi(multi_piano_roll, sf=50)
midi.write("prova.mid")
"""




# multi_piano_roll = np.round(np.load("output_test.npy") * 127.0, decimals=0).astype(np.int)
# song_length = multi_piano_roll.shape[1]

# multi_piano_roll = np.reshape(multi_piano_roll.transpose([0, 2, 1]), (4, 128, song_length))

# midi = piano_roll_to_pretty_midi(multi_piano_roll)
# midi.write("output_song.mid")

