import torch

### DATASET PARAMETERS ###
BEAT_RESOLUTION = 4
MEASURE_LENGTH = 4 * BEAT_RESOLUTION
LOWEST_PITCH = 24
N_PITCHES = 72
N_MEASURES_FOR_SAMPLE = 4
N_SAMPLES_PER_SONG = 8
N_TRACKS = 5
PROGRAMS = [0, 0, 25, 33, 48]
IS_DRUMS = [True, False, False, False, False]
TRACK_NAMES = ['Drums', 'Piano', 'Guitar', 'Bass', 'Strings']
TEMPO = 100

### TRAINING PARAMETERS ###
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_DIM = 128
GAMMA = 10.0
BATCH_SIZE = 64 // N_SAMPLES_PER_SONG
EPOCHS = 1000
LOG_INTERVAL = 10
GENERATOR_UPDATE_INTERVAL = 5
TEST = False

### CHECKPOINT PATHS ###
COLAB_TRAINING = True

CHECKPOINT_PATH = "checkpoints/checkpoint_" if not COLAB_TRAINING else "../drive/My Drive/MusicGAN/checkpoints/checkpoint_"
WRITER_PATH = "" if not COLAB_TRAINING else "../drive/My Drive/MusicGAN/runs/long_run"
IMG_SAVE_PATH = "imgs" if not COLAB_TRAINING else "../drive/My Drive/MusicGAN/imgs"



