import torch
from torch import nn
from pathlib import Path
from models import OMREncoder, OMRDecoder, ViTOMR, OMRLoss
from datasets import GrandStaffLMXDataset, GrandStaffOMRTrainWrapper, OlimpicDataset
from utils import GRAND_STAFF_ROOT_DIR, OLIMPIC_SCANNED_ROOT_DIR, OLIMPIC_SYNTHETIC_ROOT_DIR, LMX_BOS_TOKEN, LMX_EOS_TOKEN
from utils import DynamicResize, cosine_anneal_with_warmup
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.transforms import v2
from pre_train import PATCH_SIZE, PE_MAX_HEIGHT, PE_MAX_WIDTH, ragged_collate_fn
import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np

MODEL_DIR_PATH = Path("omr_train")
CHECKPOINTS_DIR_PATH = MODEL_DIR_PATH / "checkpoints"
STATS_DIR_PATH = MODEL_DIR_PATH / "stats"

PRETRAINED_MAE_STATE_DICT_PATH = "mae_pre_train/pretrained_mae.pth"
MAX_IMG_SEQ_LEN = 512 # for DynamicResize
MAX_LMX_SEQ_LEN = 512 # in tokens, max lmx token seuqence length to support
LMX_VOCAB_PATH = "lmx_vocab.txt"

AUGMENTATION_P = 0.3
NUM_WORKERS = 24

EPOCHS = 30
CHECKPOINT_FREQ = 5
BASE_LR = 2e-4
MIN_LR = 1e-6
ADAMW_BETAS = (0.9, 0.95)
ADAMW_WEIGHT_DECAY = 0.01
WARMUP_EPOCHS = 2 # step scheduler per-batch since doing so little epochs
BATCH_SIZE = 64

# additional regularization: dropout of 0.1 in decoder, label smoothing of 0.1

# transform that appends beginning/end of sequence tokens, converts tokens to vocab indices
class PrepareLMXSequence(nn.Module):
    def forward(self, lmx: str):
        tokens = [LMX_BOS_TOKEN]
        tokens.append(lmx.strip().split())
        tokens.append(LMX_EOS_TOKEN)

if __name__ == "__main__":
    encoder = OMREncoder(PATCH_SIZE, PE_MAX_HEIGHT, PE_MAX_WIDTH)
    print(f"Encoder set up with patch size {PATCH_SIZE}, pe grid of {PE_MAX_HEIGHT} x {PE_MAX_WIDTH}")
    decoder = OMRDecoder(MAX_LMX_SEQ_LEN, LMX_VOCAB_PATH)
    print(f"Decoder set up with max lmx sequence length {MAX_LMX_SEQ_LEN}, vocab file {LMX_VOCAB_PATH}")
    pretrained_mae_state_dict = torch.load(PRETRAINED_MAE_STATE_DICT_PATH)
    print(f"Loaded pretrained mae state dict from {PRETRAINED_MAE_STATE_DICT_PATH}")
    vitomr = ViTOMR(encoder, pretrained_mae_state_dict, decoder)
    print("set up ViTOMR model")

    base_img_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        DynamicResize(PATCH_SIZE, MAX_IMG_SEQ_LEN, PE_MAX_HEIGHT, PE_MAX_WIDTH, False)
    ])

    base_lmx_transform = PrepareLMXSequence()

    grand_staff = GrandStaffLMXDataset(GRAND_STAFF_ROOT_DIR, "samples.train.txt")

# TODO: implement per-batch version of scheduler in utils
# training loops for omr training
# add omr mode to eval_model.py, separate mae testing
# implement omr autoregressive inference

"""
Chat's recommended recipe
🔧 Core Training Hyperparameters
Hyperparameter	Value / Range	Notes
Epochs	10–30	Depending on dataset size; early stop on val loss.
Batch size	32–128	As large as fits in GPU memory; larger = more stable.
Optimizer	AdamW	Standard for transformers.
Initial LR	1e-4	You can go slightly higher if encoder is frozen.
Min LR	1e-6	For cosine schedule.
Weight decay	0.01	Typical for Transformer-based models.
Scheduler	Cosine Annealing + Warmup	Smooth convergence.
Warmup steps	5% of total steps	Critical to prevent initial divergence.
Gradient clipping	1.0	Prevent exploding gradients.
Dropout	0.1	Only in decoder or adapters.
Label smoothing	0.0–0.1	Optional. Slight smoothing can improve generalization.
"""