import numpy as np

DEVICE = 'cuda:0'

# parameters for input representation
DEFAULT_POS_PER_QUARTER = 12
DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 32+1, dtype=np.int)
DEFAULT_DURATION_BINS = np.sort(np.concatenate([
  np.arange(1, 13), # smallest possible units up to 1 quarter
  np.arange(12, 24, 3)[1:], # 16th notes up to 1 bar
  np.arange(13, 24, 4)[1:], # triplets up to 1 bar
  np.arange(24, 48, 6), # 8th notes up to 2 bars
  np.arange(48, 4*48, 12), # quarter notes up to 8 bars
  np.arange(4*48, 16*48+1, 24) # half notes up to 16 bars
]))
#[  1   2   3   4   5   6   7   8   9  10  11  12  15  17  18  21  21  24
#  30  36  42  48  60  72  84  96 108 120 132 144 156 168 180 192 216 240
# 264 288 312 336 360 384 408 432 456 480 504 528 552 576 600 624 648 672
# 696 720 744 768]
DEFAULT_TEMPO_BINS = np.linspace(0, 240, 32+1, dtype=np.int)
#[  0   7  15  22  30  37  45  52  60  67  75  82  90  97 105 112 120 127
# 135 142 150 157 165 172 180 187 195 202 210 217 225 232 240]
#DEFAULT_TEMPO_BINS = np.linspace(0, 240, 240+1, dtype=np.int)
DEFAULT_NOTE_DENSITY_BINS = np.linspace(0, 12, 32+1)
DEFAULT_MEAN_VELOCITY_BINS = np.linspace(0, 128, 32+1)
DEFAULT_MEAN_PITCH_BINS = np.linspace(0, 128, 32+1)
DEFAULT_MEAN_DURATION_BINS = np.logspace(0, 7, 32+1, base=2) # log space between 1 and 128 positions (~2.5 bars)
DEFAULT_RHYTHM_INTENSITY = [0.2, 0.25, 0.32, 0.38, 0.44, 0.5, 0.63]
DEFAULT_POLYPHONY_BINS = [2.63, 3.06, 3.50, 4.00, 4.63, 5.44, 6.44]

# parameters for output
DEFAULT_RESOLUTION = 480

# maximum length of a single bar is 3*4 = 12 beats
MAX_BAR_LENGTH = 3
# maximum number of bars in a piece is 512 (this covers almost all sequences)
MAX_N_BARS = 512

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'
MASK_TOKEN = '<mask>'

TIME_SIGNATURE_KEY = 'Time Signature'
BAR_KEY = 'Bar'
POSITION_KEY = 'Position'
INSTRUMENT_KEY = 'Instrument'
PITCH_KEY = 'Pitch'
VELOCITY_KEY = 'Velocity'
DURATION_KEY = 'Duration'
TEMPO_KEY = 'Tempo'
CHORD_KEY = 'Chord'

NOTE_DENSITY_KEY = 'Note Density'
MEAN_PITCH_KEY = 'Mean Pitch'
MEAN_VELOCITY_KEY = 'Mean Velocity'
MEAN_DURATION_KEY = 'Mean Duration'
RHYTHM_INTENSITY_KEY = 'Rhythm Intensity'
POLYPHONY_KEY = 'Polyphony'