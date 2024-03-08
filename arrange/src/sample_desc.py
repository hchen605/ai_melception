
import os
import glob
import time
import torch
import random
from torch.utils.data import DataLoader

from models.vae import VqVaeModule
from models.seq2seq import Seq2SeqModule
from datasets import MidiDataset, SeqCollator
from utils import medley_iterator
from input_representation import remi2midi

MODEL = os.getenv('MODEL', '')

#ROOT_DIR = os.getenv('ROOT_DIR', './lmd_full')
ROOT_DIR = os.getenv('ROOT_DIR', './arrange/data')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './arrange/samples')
DESC_DIR = os.getenv('DESC_DIR', './arrange/desc')
FILE = os.getenv('FILE','./arrange/data/input.mid')
MAX_N_FILES = int(float(os.getenv('MAX_N_FILES', -1)))
MAX_ITER = int(os.getenv('MAX_ITER', 16_000))
MAX_BARS = int(os.getenv('MAX_BARS', 32))

MAKE_MEDLEYS = os.getenv('MAKE_MEDLEYS', 'False') == 'True'
N_MEDLEY_PIECES = int(os.getenv('N_MEDLEY_PIECES', 2))
N_MEDLEY_BARS = int(os.getenv('N_MEDLEY_BARS', 16))
  
CHECKPOINT = os.getenv('CHECKPOINT', None)
VAE_CHECKPOINT = os.getenv('VAE_CHECKPOINT', None)
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 1))
VERBOSE = int(os.getenv('VERBOSE', 2))

def reconstruct_sample(model, batch, 
  initial_context=1, 
  output_dir=None, 
  max_iter=-1, 
  max_bars=-1,
  verbose=0,
):
  batch_size, seq_len = batch['input_ids'].shape[:2]

  batch_ = { key: batch[key][:, :initial_context] for key in ['input_ids', 'bar_ids', 'position_ids'] }
  if model.description_flavor in ['description', 'both']:
    batch_['description'] = batch['description']
    batch_['desc_bar_ids'] = batch['desc_bar_ids']
  if model.description_flavor in ['latent', 'both']:
    batch_['latents'] = batch['latents']

  print('------ Input description generating ... --------')
  #print(batch['desc_events'])
  
  #file_path = 'arrange/desc/description.txt'
  if DESC_DIR:
    os.makedirs(DESC_DIR, exist_ok=True)
  file_path = DESC_DIR + '/description.txt'
  
  prefix_condition = 'Bar_'
  with open(file_path, 'w') as file:
    # Write each string from the list to the file
    for i in range(len(batch['desc_events'][0]) - 1):
      item = batch['desc_events'][0][i]
      if batch['desc_events'][0][i+1].startswith(prefix_condition):
          file.write("%s,\n" % item)
      else:
          file.write("%s," % item)
    file.write("%s" % batch['desc_events'][0][-1])
  print('------ Finished --------')
  #return events


def main():
  if MAKE_MEDLEYS:
    max_bars = N_MEDLEY_PIECES * N_MEDLEY_BARS
  else:
    max_bars = MAX_BARS

  if OUTPUT_DIR:
    params = []
    if MAKE_MEDLEYS:
      params.append(f"n_pieces={N_MEDLEY_PIECES}")
      params.append(f"n_bars={N_MEDLEY_BARS}")
    if MAX_ITER > 0:
      params.append(f"max_iter={MAX_ITER}")
    if MAX_BARS > 0:
      params.append(f"max_bars={MAX_BARS}")
    output_dir = os.path.join(OUTPUT_DIR, MODEL, ','.join(params))
  else:
    raise ValueError("OUTPUT_DIR must be specified.")

  print(f"Saving generated files to: {output_dir}")

  if VAE_CHECKPOINT:
    vae_module = VqVaeModule.load_from_checkpoint(VAE_CHECKPOINT)
    vae_module.cpu()
  else:
    vae_module = None

  model = Seq2SeqModule.load_from_checkpoint(CHECKPOINT)
  model.freeze()
  model.eval()

  print('------ Load MIDI --------')

  #midi_files = glob.glob(os.path.join(ROOT_DIR, '**/*.mid'), recursive=True)
  #midi_files = glob.glob(ROOT_DIR + '/*.mid', recursive=True)
  #midi_files = [ROOT_DIR + '/Honestly_Piano_12.midi']
  midi_files = [FILE]
  #print(midi_files)

  if MAX_N_FILES > 0:
    midi_files = midi_files[:MAX_N_FILES]

  description_options = None
  if MODEL in ['figaro-no-inst', 'figaro-no-chord', 'figaro-no-meta']:
    description_options = model.description_options

  dataset = MidiDataset(
    midi_files,
    max_len=-1,
    description_flavor=model.description_flavor,
    description_options=description_options,
    max_bars=model.context_size,
    vae_module=vae_module
  )

  print('------ Read event/description --------')

  start_time = time.time()
  coll = SeqCollator(context_size=-1)
  dl = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=coll)

  if MAKE_MEDLEYS:
    dl = medley_iterator(dl, 
      n_pieces=N_MEDLEY_BARS, 
      n_bars=N_MEDLEY_BARS, 
      description_flavor=model.description_flavor
    )
  
  with torch.no_grad():
    for batch in dl:
      reconstruct_sample(model, batch, 
        output_dir=output_dir, 
        max_iter=MAX_ITER, 
        max_bars=max_bars,
        verbose=VERBOSE,
      )

if __name__ == '__main__':
  main()
