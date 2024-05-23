import os
import torch
import glob

import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from models.seq2seq import Seq2SeqModule
from models.vae import VqVaeModule
import pdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_CODES = 2048  # [VQ-VAE] Codebook size
N_GROUPS = 16  # [VQ-VAE] Number of groups to split the latent vector into before discretization
D_MODEL = 512  # Hidden size of the model
D_LATENT = 1024  # [VQ-VAE] Dimensionality of the latent space

VAE_CHECKPOINT = None

WARMUP_STEPS = 4000
MAX_STEPS = 1e20
MAX_TRAINING_STEPS = 100000  # DON'T CHANGE! Max. number of training iterations.
LR_SCHEDULE = os.getenv('LR_SCHEDULE', 'const')  # Current: sqrt
CONTEXT_SIZE = 256

ACCUMULATE_GRADS = 4 # TARGET_BATCH_SIZE//BATCH_SIZE

N_WORKERS = min(os.cpu_count(), float(os.getenv('N_WORKERS', 'inf')))  # 20
if device.type == 'cuda':
	N_WORKERS = min(N_WORKERS, 8*torch.cuda.device_count())  # (20, 8)
N_WORKERS = int(N_WORKERS)  # 8

ADD_ATTR = {'rhythm_int': False, 'polyphonic': False}

def main(args):
	### Define available models ###
	available_models = [
		'vq-vae',
		'figaro-learned',
		'figaro-expert',
		'figaro',
		'figaro-inst',
		'figaro-chord',
		'figaro-meta',
		'figaro-no-inst',
		'figaro-no-chord',
		'figaro-no-meta',
		'baseline',
	]

	assert args.model in available_models, f'unknown MODEL: {args.model}'

	### Load in all midi files ###
	midi_files = glob.glob(os.path.join(args.root_dir, '**/*.mid'), recursive=True)
	if args.max_n_file > 0:
		midi_files = midi_files[:args.max_n_file]

	if len(midi_files) == 0:
		print(f"WARNING: No MIDI files were found at '{args.root_dir}'. Did you download the dataset to the right location?")
		exit()

	MAX_CONTEXT = min(1024, CONTEXT_SIZE)  # 256

	#if args.model in ['figaro-learned', 'figaro'] and VAE_CHECKPOINT:
	#	vae_module = VqVaeModule.load_from_checkpoint(checkpoint_path=VAE_CHECKPOINT)
	#	vae_module.cpu()
	#	vae_module.freeze()
	#	vae_module.eval()
	#else:
	vae_module = None


	### Create and train model ###
	# load model from checkpoint if available
	if args.checkpoint:
		print('load checkpoint: {}'.format(args.checkpoint))
		model_class = {
			'vq-vae': VqVaeModule,
			'figaro-learned': Seq2SeqModule,
			'figaro-expert': Seq2SeqModule,
			'figaro': Seq2SeqModule,
			'figaro-inst': Seq2SeqModule,
			'figaro-chord': Seq2SeqModule,
			'figaro-meta': Seq2SeqModule,
			'figaro-no-inst': Seq2SeqModule,
			'figaro-no-chord': Seq2SeqModule,
			'figaro-no-meta': Seq2SeqModule,
			'baseline': Seq2SeqModule,
		}[args.model]
		model = model_class.load_from_checkpoint(checkpoint_path=args.checkpoint)

	else:
		seq2seq_kwargs = {
			'encoder_layers': 4,
			'decoder_layers': 6,
			'num_attention_heads': 8,
			'intermediate_size': 2048,
			'd_model': D_MODEL,  # hidden_size = 512
			'context_size': MAX_CONTEXT,  # 256
			'lr': args.lr,
			'warmup_steps': WARMUP_STEPS,
			'max_steps': MAX_STEPS,
		}
		dec_kwargs = { **seq2seq_kwargs }
		dec_kwargs['encoder_layers'] = 0

		# use lambda functions for lazy initialization
		model = {
			'vq-vae': lambda: VqVaeModule(
				encoder_layers=4,
				decoder_layers=6,
				encoder_ffn_dim=2048,
				decoder_ffn_dim=2048,
				n_codes=N_CODES, 
				n_groups=N_GROUPS, 
				context_size=MAX_CONTEXT,
				lr=args.lr,
				lr_schedule=LR_SCHEDULE,
				warmup_steps=WARMUP_STEPS,
				max_steps=MAX_STEPS,
				d_model=D_MODEL,
				d_latent=D_LATENT,
			),
			'figaro-learned': lambda: Seq2SeqModule(
				description_flavor='latent',
				n_codes=vae_module.n_codes,
				n_groups=vae_module.n_groups,
				d_latent=vae_module.d_latent,
				**seq2seq_kwargs
			),
			'figaro': lambda: Seq2SeqModule(
				description_flavor='both',
				n_codes=vae_module.n_codes,
				n_groups=vae_module.n_groups,
				d_latent=vae_module.d_latent,
				**seq2seq_kwargs
			),
			'figaro-expert': lambda: Seq2SeqModule(
				description_flavor='description',
				**seq2seq_kwargs
			),
			'figaro-no-meta': lambda: Seq2SeqModule(
				description_flavor='description',
				description_options={'instruments': True, 'chords': True, 'meta': False, 'rhyt':True, 'poly':True},
				**seq2seq_kwargs
			),
			'figaro-no-inst': lambda: Seq2SeqModule(
				description_flavor='description',
				description_options={'instruments': False, 'chords': True, 'meta': True, 'rhyt':True, 'poly':True},
				**seq2seq_kwargs
			),
			'figaro-no-chord': lambda: Seq2SeqModule(
				description_flavor='description',
				description_options={'instruments': True, 'chords': False, 'meta': True, 'rhyt':True, 'poly':True},
				**seq2seq_kwargs
			),
			'figaro-no-rhythm': lambda: Seq2SeqModule(
				description_flavor='description',
				description_options={'instruments': True, 'chords': True, 'meta': True, 'rhyt':False, 'poly':True},
				**seq2seq_kwargs
			),
			'figaro-no-poly': lambda: Seq2SeqModule(
				description_flavor='description',
				description_options={'instruments': True, 'chords': False, 'meta': True, 'rhyt':True, 'poly':False},
				**seq2seq_kwargs
			),
			'baseline': lambda: Seq2SeqModule(
				description_flavor='none',
				**dec_kwargs
			),
		}[args.model]()

	### Create data loaders ###
	datamodule = model.get_datamodule(
		midi_files,
		vae_module=vae_module,
		batch_size=args.batch, 
		num_workers=N_WORKERS, 
		pin_memory=True
	)

	checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
		monitor='valid_loss',
		dirpath=os.path.join(args.output_dir, args.exp_name), # MODEL
		filename='{step}-{valid_loss:.2f}',
		save_last=True,
		save_top_k=2,
		every_n_train_steps= 1000
	)

	lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

	trainer = pl.Trainer(
		devices=1,
		accelerator='gpu',
		strategy='dp',
		profiler='simple',
		callbacks=[checkpoint_callback, lr_monitor, 
					pl.callbacks.stochastic_weight_avg.StochasticWeightAveraging(swa_lrs=1e-2),
					],
		max_epochs=args.epoch,
		max_steps=MAX_TRAINING_STEPS,
		log_every_n_steps= 100, #max(100, min(25*ACCUMULATE_GRADS, 200)),
		val_check_interval= 1000, #max(500, min(300*ACCUMULATE_GRADS, 1000)),  #  = 1000 / 500 --> runs validation set
		limit_val_batches=64,
		auto_scale_batch_size=False,
		auto_lr_find=False,
		accumulate_grad_batches=ACCUMULATE_GRADS,
		gradient_clip_val=1.0, 
		detect_anomaly=True,
		resume_from_checkpoint=args.checkpoint,
		#enable_checkpointing=True,
		#enable_progress_bar=True,
		#enable_model_summary=True
	)

	trainer.fit(model, datamodule)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--exp_name', type=str, require=True, help='folder name by date')
	parser.add_argument('--root_dir', type=str, default='/ssddata2/joann/lmd_full', help='root dir for training data')
	parser.add_argument('--output_dir', type=str, default='./results', help='output directory for training checkpoints')
	parser.add_argument('--max_n_file', type=int, default=-1, help='max number of midi files to process')
	parser.add_argument('--model', type=str, required=True, help='model name')
	parser.add_argument('--checkpoint', type=str, default=None, help='path for checkpoint to use')
	parser.add_argument('--batch', type=int, default=1, help='batch size')  # 128
	parser.add_argument('--target_batch', type=int, default=4, help='Number of samples in each backward step')
	parser.add_argument('--epoch', type=int, default=64, help='Max. number of training epochs')
	parser.add_argument('--lr', type=int, default=1e-4, help='Learning rate')
	args = parser.parse_args()
	main(args)