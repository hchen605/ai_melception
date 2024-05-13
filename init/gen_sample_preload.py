import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["KMP_DUPLICATE_LIB_OK"] ="TRUE"
import numpy as np
import pretty_midi as pyd
#import IPython.display as ipd
from arrangement_utils import *
import warnings
warnings.filterwarnings("ignore")

DATA_FILE_ROOT = './data_file_dir/'
DEVICE = 'cuda:0'

# load data, init model
piano_arranger, orchestrator, piano_texture, band_prompt = load_premise_preload(DATA_FILE_ROOT, DEVICE)
#

"""Set input lead sheet"""
#SONG_NAME, SEGMENTATION, PICKUP_BEAT, TEMPO = 'Castles in the Air', 'A8A8B8B8', 1, 100   #1 beat in the pick-up measure
#SONG_NAME, SEGMENTATION, PICKUP_BEAT, TEMPO = 'Jingle Bells', 'A8B8A8', 0, 100
#SONG_NAME, SEGMENTATION, PICKUP_BEAT, TEMPO = 'Sally Garden', 'A4A4B4A4', 0, 75
#SONG_NAME, SEGMENTATION, PICKUP_BEAT, TEMPO = 'Auld Lang Syne', 'A8B8A8B8', 1, 80
SONG_NAME, SEGMENTATION, PICKUP_BEAT, TEMPO = 'Honestly', 'A4A4B4', 0, 70

"""Set texture pre-filtering for piano arrangement (default random)"""
RHTHM_DENSITY = np.random.randint(3, 5)  # 4
VOICE_NUMBER = np.random.randint(2, 5)
PREFILTER = (RHTHM_DENSITY, VOICE_NUMBER)

"""Set if use a 2-bar prompt for full-band arrangement (default True)"""
USE_PROMPT = True
midi_path = 'Honestly_Piano_12.midi'
lead_sheet = read_lead_sheet('./demo', SONG_NAME, SEGMENTATION, PICKUP_BEAT, midi_path)

#"""have a quick listen to the lead sheet"""
print(' -- Load Piano lead sheet MIDI ready -- ')
midi_piano, acc_piano = piano_arrangement(*lead_sheet, *piano_texture, piano_arranger, PREFILTER, TEMPO)
arrg_piano = f'./demo/{SONG_NAME}/arrangement_piano.mid'
midi_piano.write(arrg_piano)

func_prompt = prompt_sampling(acc_piano, *band_prompt, DEVICE)
print(' -- func prompt ready -- ')

if USE_PROMPT:  #condition with both instrument set and track function of the first 2 bar
    midi_band = orchestration(acc_piano, None, *func_prompt, orchestrator, DEVICE, blur=.25, p=.05, t=6, tempo=TEMPO)
else: #condition with instrument set only
    instruments, track_function = func_prompt
    midi_band = orchestration(acc_piano, None, instruments, None, orchestrator, DEVICE, blur=.25, p=.05, t=8, tempo=TEMPO)
mel_track = pyd.Instrument(program=0, is_drum=False, name='melody')
mel_track.notes = midi_piano.instruments[0].notes
midi_band.instruments.append(mel_track)

arrg_band = f'./demo/{SONG_NAME}/arrangement_band.mid'
midi_band.write(arrg_band)

print(' -- Full band MIDI ready -- ')