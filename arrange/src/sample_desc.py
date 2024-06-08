import os
#import time
import torch
from torch.utils.data import DataLoader

from models.vae import VqVaeModule
from models.seq2seq import Seq2SeqModule
from datasets import MidiDataset, SeqCollator
from utils import medley_iterator
import argparse
import pdb

MAX_ITER = 16000
MAKE_MEDLEYS = False
N_MEDLEY_PIECES, N_MEDLEY_BARS = 2, 16
VAE_CHECKPOINT = None
VERBOSE = 2

def reconstruct_sample(model, batch, desc_dir, initial_context=1, max_iter=-1, max_bars=-1,verbose=0):
	#batch_size, seq_len = batch['input_ids'].shape[:2]

	batch_ = {key: batch[key][:, :initial_context] for key in ['input_ids', 'bar_ids', 'position_ids']}
	if model.description_flavor in ['description', 'both']:
		batch_['description'] = batch['description']  # all descriptions
		batch_['desc_bar_ids'] = batch['desc_bar_ids'] # aligned bar ids for descriptions
	if model.description_flavor in ['latent', 'both']:
		batch_['latents'] = batch['latents']

	print('------ Input description generating ... --------')
	
	if desc_dir:
		os.makedirs(desc_dir, exist_ok=True)
	file_path = os.path.join(desc_dir, 'description.txt')
	
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


def main(args):
	EXP_NAME, MODEL, DESC_DIR, FILE, MAX_N_FILES, MAX_BARS, CHECKPOINT, BATCH_SIZE = vars(args).values() #OUTPUT_DIR, 
	
	if MAKE_MEDLEYS:  # default False
		max_bars = N_MEDLEY_PIECES * N_MEDLEY_BARS
	else:
		max_bars = MAX_BARS

	#if OUTPUT_DIR:
	#	params = []
	#	if MAKE_MEDLEYS:
	#		params.append(f"n_pieces={N_MEDLEY_PIECES}")
	#		params.append(f"n_bars={N_MEDLEY_BARS}")
	#	if MAX_ITER > 0:  # max_iter = 16000
	#		params.append(f"max_iter={MAX_ITER}")  # params = ['max_iter=16000']
	#	if MAX_BARS > 0:  # max_bars = 32
	#		params.append(f"max_bars={MAX_BARS}")  # params = ['max_iter=16000', 'max_bars=32']
	#	output_dir = os.path.join(OUTPUT_DIR, EXP_NAME) # os.path.join(OUTPUT_DIR, MODEL, ','.join(params))
	#else:
	#	raise ValueError("OUTPUT_DIR must be specified.")

	#print(f"Saving generated files to: {output_dir}")

	if VAE_CHECKPOINT:  # default False
		vae_module = VqVaeModule.load_from_checkpoint(VAE_CHECKPOINT)
		vae_module.cpu()
	else:
		vae_module = None

	model = Seq2SeqModule.load_from_checkpoint(CHECKPOINT, map_location=lambda storage, loc: storage)
	model.freeze()
	model.eval()

	print('------ Load MIDI --------')
	midi_files = [FILE]

	if MAX_N_FILES > 0:  # default -1
		midi_files = midi_files[:MAX_N_FILES]

	description_options = None
	if MODEL in ['figaro-no-inst', 'figaro-no-chord', 'figaro-no-meta']:
		description_options = model.description_options

	dataset = MidiDataset(
						midi_files,
						max_len=-1,
						description_flavor=model.description_flavor,  # description
						description_options=description_options,  # none
						max_bars=model.context_size,  # 256+8
						vae_module=vae_module
					)

	print('------ Read event/description --------')
	#start_time = time.time()
	coll = SeqCollator(context_size=-1)
	dl = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=coll)
	
	if MAKE_MEDLEYS:
		dl = medley_iterator(
							dl, 
							n_pieces=N_MEDLEY_BARS,  ## N_MEDLEY_PIECES??
							n_bars=N_MEDLEY_BARS, 
							description_flavor=model.description_flavor
						)
	
	with torch.no_grad():
		for batch in dl:
			reconstruct_sample(
							model, 
							batch, 
							desc_dir=os.path.join(DESC_DIR),
							#output_dir=output_dir, 
							max_iter=MAX_ITER, 
							max_bars=max_bars,
							verbose=VERBOSE
						)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--exp_name', type=str, default='0523_rhythm', help='folder name by date')
	parser.add_argument('--model', type=str, default='figaro-expert', help='model name')
	#parser.add_argument('--output_dir', type=str, default='./arrange/samples', help='output directory for generated samples')
	parser.add_argument('--desc_dir', type=str, default='../proto-demo/frontend/public/from_back/Honestly', help='output directory for descriptions')
	parser.add_argument('--midi_file', type=str, default='../proto-demo/frontend/public/from_back/Honestly/Honestly_Piano_12.midi', help='file path for midi')
	parser.add_argument('--max_n_file', type=int, default=-1, help='max number of midi files to process')
	parser.add_argument('--max_bars', type=int, default=32, help='max number of bars')
	parser.add_argument('--checkpoint', type=str, default='arrange/results/0523_rhythm/step=41000-valid_loss=1.13.ckpt', help='path for checkpoint to use')
	parser.add_argument('--batch', type=int, default=1, help='batch size')
	args = parser.parse_args()
	main(args)
