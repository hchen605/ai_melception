import os
os.environ['CUDA_VISIBLE_DEVICES']= '0'
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import Query_and_reArrange
from dataset import Slakh2100_Pop909_Dataset, collate_fn_inference, EMBED_PROGRAM_MAPPING
SLAKH_CLASS_MAPPING = {v: k for k, v in EMBED_PROGRAM_MAPPING.items()}
from utils.format_convert import matrix2midi_with_dynamics, dataitem2midi
from utils.inferring import mixture_function_prior, search_reference, velocity_adaption
import datetime
import warnings
warnings.filterwarnings("ignore")

POP909_DIR = "./data/POP909"
SLAKH2100_DIR = "./data/Slakh2100"
SAVE_DIR = './demo'

SAMPLE_BAR_LEN = 8

MODEL_DIR = "./checkpoints/Q&A_epoch_029.pt"
DEVICE = 'cuda:0'
model = Query_and_reArrange(name='inference_model', device=DEVICE, trf_layers=2)
model.load_state_dict(torch.load(MODEL_DIR, map_location='cpu'))
model.to(DEVICE)
model.eval()

## Orchestration

# load piano dataset. A piano piece x is the donor of content.
x_set = Slakh2100_Pop909_Dataset(None, POP909_DIR, 16*SAMPLE_BAR_LEN, debug_mode=True, split='validation', mode='inference', with_dynamics=True)
# load multi-track dataset. A multi-track piece y is the donor of style.
y_set = Slakh2100_Pop909_Dataset(SLAKH2100_DIR, None, 16*SAMPLE_BAR_LEN, debug_mode=True, split='validation', mode='inference', with_dynamics=True)
# Prepare for the heuristic sampling of y
y_set_loader = DataLoader(y_set, batch_size=1, shuffle=False, collate_fn=lambda b: collate_fn_inference(b, DEVICE))
y_prior_set = mixture_function_prior(y_set_loader)

# get a random x sample
IDX = np.random.randint(len(x_set))
x = x_set.__getitem__(IDX)
(x_mix, x_instr, x_fp, x_ft), x_dyn, x_dir = collate_fn_inference(batch = [(x)], device = DEVICE)
# save x
save_path = os.path.join(SAVE_DIR, f"orchestration-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')[2:]}")
if not os.path.exists(save_path):
    os.makedirs(save_path)
x_recon = dataitem2midi(*x, SLAKH_CLASS_MAPPING)
x_recon.write(os.path.join(save_path, '01_source.mid'))
print(f'saved to {save_path}.')

# heuristic sampling for y (i.e., Equation (8) in the paper)
y_anchor = search_reference(x_fp, x_ft, y_prior_set)
y = y_set.__getitem__(y_anchor)
(y_mix, y_instr, y_fp, y_ft), y_dyn, y_dir = collate_fn_inference(batch=[(y)], device=DEVICE)
# exchange x's and y's melody track function in order to preserve the theme melody after rearrangement.
x_mel = 0
y_mel = np.argmax(np.mean(np.ma.masked_equal(y_dyn[..., 0], value=0), axis=(1, 2))) #pick the track with highest velocity
y_fp[:, y_mel] = x_fp[:, x_mel]
y_ft[:, y_mel] = x_ft[:, x_mel]
#save y
y_recon = dataitem2midi(*y, SLAKH_CLASS_MAPPING)
y_recon.write(os.path.join(save_path, '02_reference.mid'))

# Q&A model inference
output = model.inference(x_mix, y_instr, y_fp, y_ft, mel_id=y_mel)
# apply y's dynamics to the rearrangement result
velocity = velocity_adaption(y_dyn[..., 0], output, y_mel)
cc = y_dyn[..., 1]
output = np.stack([output, velocity, cc], axis=-1)
# reconstruct MIDI
midi_recon = matrix2midi_with_dynamics(
    matrices=output, 
    programs=[SLAKH_CLASS_MAPPING[item.item()] for item in y_instr[0]], 
    init_tempo=100)
midi_recon.write(os.path.join(save_path, '03_target.mid'))
print(f'saved to {save_path}.')

