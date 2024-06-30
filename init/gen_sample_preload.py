import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']= '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import pretty_midi as pyd
from midi2audio import FluidSynth
#import IPython.display as ipd
from arrangement_utils import *
import argparse
import warnings
warnings.filterwarnings("ignore")

#DATA_FILE_ROOT = './data_file_dir/'
DATA_FILE_ROOT = '../../ai_melception/init/data_file_dir/'
#DEVICE = 'cuda:0'
# Check if CUDA is available and get the device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# load data, init model
piano_arranger, orchestrator, piano_texture, band_prompt = load_premise_preload(DATA_FILE_ROOT, DEVICE)

#--
# Argument parser
parser = argparse.ArgumentParser(description='MIDI Arrangement Configuration')
parser.add_argument('--midi_path', type=str, default='Honestly_Piano_12.midi', help='Path to the input MIDI file')
parser.add_argument('--rhythm', type=int, default=3, help='Rhythm density for the piano arrangement')
parser.add_argument('--polyphony', type=int, default=3, help='Number of voices for the piano arrangement')
parser.add_argument('--segment', type=str, default='A4A4B4', help='segmentation')
parser.add_argument('--tempo', type=int, default=70, help='Tempo for the arrangement')
parser.add_argument('--output_piano', type=str, required=True, help='Output path for piano arrangement')
parser.add_argument('--output_band', type=str, required=True, help='Output path for band arrangement')

args = parser.parse_args()

"""Set input lead sheet"""
#SONG_NAME, SEGMENTATION, PICKUP_BEAT, TEMPO = 'Castles in the Air', 'A8A8B8B8', 1, 100   #1 beat in the pick-up measure
#SONG_NAME, SEGMENTATION, PICKUP_BEAT, TEMPO = 'Jingle Bells', 'A8B8A8', 0, 100
#SONG_NAME, SEGMENTATION, PICKUP_BEAT, TEMPO = 'Sally Garden', 'A4A4B4A4', 0, 75
#SONG_NAME, SEGMENTATION, PICKUP_BEAT, TEMPO = 'Auld Lang Syne', 'A8B8A8B8', 1, 80
#SONG_NAME, SEGMENTATION, PICKUP_BEAT, TEMPO = 'Honestly', 'A4A4B4', 0, 70

"""Set texture pre-filtering for piano arrangement (default random)"""

PICKUP_BEAT = 0
MIDI_PATH = args.midi_path
RHTHM_DENSITY = args.rhythm
VOICE_NUMBER = args.polyphony 
PREFILTER = (RHTHM_DENSITY, VOICE_NUMBER)
TEMPO = args.tempo
SEGMENTATION = args.segment


"""Set if use a 2-bar prompt for full-band arrangement (default True)"""
USE_PROMPT = True
#midi_path = f'./demo/{SONG_NAME}/Honestly_Piano_12.midi'
#midi_path = 'Honestly_Piano_12.midi'
#midi_in = pyd.PrettyMIDI(os.path.join('./demo', SONG_NAME, midi_path))
#lead_sheet = read_lead_sheet('./demo', SONG_NAME, SEGMENTATION, PICKUP_BEAT, midi_path)
lead_sheet = read_lead_sheet(MIDI_PATH, SEGMENTATION, PICKUP_BEAT)

#"""have a quick listen to the lead sheet"""

print(' -- Load Piano lead sheet MIDI ready -- ')

midi_piano, acc_piano = piano_arrangement(*lead_sheet, *piano_texture, piano_arranger, PREFILTER, TEMPO)
midi_path = f'./demo/arrangement_piano.mid'
midi_piano.write(args.output_piano)

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

#midi_band.instruments.append(midi_in.instruments[0])

midi_path = f'./demo/arrangement_band.mid'
midi_band.write(args.output_band)

print(' -- Full band MIDI ready -- ')