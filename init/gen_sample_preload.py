import os
os.environ["KMP_DUPLICATE_LIB_OK"] ="TRUE"
import numpy as np
import pretty_midi as pm
#import IPython.display as ipd
from arrangement_utils import *
import argparse
import warnings
warnings.filterwarnings("ignore")

SONG_BOOK = {'Castles in the Air': ['A8A8B8B8', 1, 100],  #1 beat in the pick-up measure
            'Jingle Bells': ['A8B8A8', 0, 100],
            'Sally Garden': ['A4A4B4A4', 0, 75],
            'Auld Lang Syne': ['A8B8A8B8', 1, 80],
            'Honestly': ['A4A4B4', 0, 70, 'Honestly_Piano_12.midi']}

def main(args):
    DATA_FILE_ROOT = args.data_root
    DEMO_ROOT = './demo'
    DEVICE = args.device if torch.cuda.is_available() else 'cpu'

    # load data, init model
    piano_arranger, orchestrator, piano_texture, band_prompt = load_premise_preload(DATA_FILE_ROOT, DEVICE)
    #

    """Set input lead sheet"""
    SONG_NAME, [SEGMENTATION, PICKUP_BEAT, TEMPO, MIDI_PATH] = args.input_leadsheet, SONG_BOOK[args.input_leadsheet]

    """Set texture pre-filtering for piano arrangement (default random)"""
    RHTHM_DENSITY = np.random.randint(3, 5)  # 4
    VOICE_NUMBER = np.random.randint(2, 5)
    PREFILTER = (RHTHM_DENSITY, VOICE_NUMBER)

    """Set if use a 2-bar prompt for full-band arrangement (default True)"""
    USE_PROMPT = True
    lead_sheet = read_lead_sheet(DEMO_ROOT, SONG_NAME, SEGMENTATION, PICKUP_BEAT, MIDI_PATH)
    # lead_sheet --> (LEADSHEET, CHORD_TABLE, melody_queries, query_phrases)

    #"""have a quick listen to the lead sheet"""
    print(' -- Load Piano lead sheet MIDI ready -- ')
    # piano_texture --> (acc_pool, edge_weights, texture_filter)
    midi_piano, acc_piano = piano_arrangement(*lead_sheet, *piano_texture, piano_arranger, PREFILTER, TEMPO)
    arrg_path = os.path.join(DEMO_ROOT, SONG_NAME, 'arrangement_piano.midi')
    midi_piano.write(arrg_path)

    func_prompt = prompt_sampling(acc_piano, *band_prompt, DEVICE)
    print(' -- func prompt ready -- ')

    if USE_PROMPT:  #condition with both instrument set and track function of the first 2 bar
        midi_band = orchestration(acc_piano, None, *func_prompt, orchestrator, DEVICE, blur=.25, p=.05, t=6, tempo=TEMPO)
    else: #condition with instrument set only
        instruments, _ = func_prompt  # instruments, track_function
        midi_band = orchestration(acc_piano, None, instruments, None, orchestrator, DEVICE, blur=.25, p=.05, t=8, tempo=TEMPO)
    mel_track = pm.Instrument(program=0, is_drum=False, name='melody')
    mel_track.notes = midi_piano.instruments[0].notes
    midi_band.instruments.append(mel_track)

    arrg_band_path = os.path.join(DEMO_ROOT, SONG_NAME, 'arrangement_band.midi')
    midi_band.write(arrg_band_path)

    print(' -- Full band MIDI ready -- ')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data_file_dir/', help='path to data files')
    parser.add_argument('--device', type=str, default='cuda:0', help='device number')
    parser.add_argument('--input_leadsheet', type=str, default='Honestly', help='input leadsheet name')
    args = parser.parse_args()
    main(args)