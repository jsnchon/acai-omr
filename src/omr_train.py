import torch
from torch import nn
from pathlib import Path
from models import OMREncoder, OMRDecoder, ViTOMR, OMRLoss
from datasets import GrandStaffLMXDataset, GrandStaffOMRTrainWrapper, OlimpicDataset
from config import GRAND_STAFF_ROOT_DIR, OLIMPIC_SCANNED_ROOT_DIR, OLIMPIC_SYNTHETIC_ROOT_DIR, LMX_BOS_TOKEN, LMX_EOS_TOKEN
from utils import DynamicResize, cosine_anneal_with_warmup
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.transforms import v2, InterpolationMode
from pre_train import PATCH_SIZE, PE_MAX_HEIGHT, PE_MAX_WIDTH, ragged_collate_fn
import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np

MODEL_DIR_PATH = Path("omr_train")
CHECKPOINTS_DIR_PATH = MODEL_DIR_PATH / "checkpoints"
STATS_DIR_PATH = MODEL_DIR_PATH / "stats"

PRETRAINED_MAE_STATE_DICT_PATH = "epoch_350_mae.pth" # "mae_pre_train/pretrained_mae.pth"
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

# TODO: implement per-batch version of scheduler in utils
# training loops for omr training
# add omr mode to eval_model.py, separate mae testing
# implement omr autoregressive inference
# transform that appends beginning/end of sequence tokens, converts tokens to vocab indices
# change mae weight path once the pretraining is done

class PrepareLMXSequence(nn.Module):
    def __init__(self, tokens_to_idxs):
        super().__init__()
        self.tokens_to_idxs = tokens_to_idxs

    def forward(self, lmx: str):
        tokens = [LMX_BOS_TOKEN]
        tokens += lmx.strip().split()
        tokens.append(LMX_EOS_TOKEN)
        return [self.tokens_to_idxs[token] for token in tokens]

def omr_train(vitomr, train_dataset, validation_dataset):
    print("Model architecture\n--------------------")
    print(vitomr)
    params_count = sum(p.numel() for p in vitomr.parameters() if p.requires_grad)
    print(f"Trainable parameters count: {params_count}") 
    print(f"Setting up DataLoaders with batch size {BATCH_SIZE}, shuffle, {NUM_WORKERS} workers, ragged collate function, pinned memory")
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=ragged_collate_fn, pin_memory=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=ragged_collate_fn, pin_memory=True)
    print(f"Dataset augmentation probability: {AUGMENTATION_P}")
    print(f"Setting up AdamW with base learning rate {BASE_LR}, betas {ADAMW_BETAS}, weight decay {ADAMW_WEIGHT_DECAY}")
    optimizer = torch.optim.AdamW(vitomr.parameters(), lr=BASE_LR, betas=ADAMW_BETAS, weight_decay=ADAMW_WEIGHT_DECAY)
    num_batches = len(train_dataloader)
    print(f"Setting up scheduler with {WARMUP_EPOCHS} warm-up epochs, {EPOCHS} total epochs, {MIN_LR} minimum learning rate, {num_batches} batches per epoch")
    scheduler = cosine_anneal_with_warmup(optimizer, WARMUP_EPOCHS, EPOCHS, MIN_LR, num_train_batches=num_batches)
    loss_fn = OMRLoss(vitomr.decoder.padding_idx)

    MODEL_DIR_PATH.mkdir(exist_ok=True)
    CHECKPOINTS_DIR_PATH.mkdir(exist_ok=True)
    STATS_DIR_PATH.mkdir(exist_ok=True)
    print(f"Created directories {MODEL_DIR_PATH}, {CHECKPOINTS_DIR_PATH}, {STATS_DIR_PATH}")


if __name__ == "__main__":
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using device {device}")

    print(f"Setting up encoder with patch size {PATCH_SIZE}, pe grid of {PE_MAX_HEIGHT} x {PE_MAX_WIDTH}")
    encoder = OMREncoder(PATCH_SIZE, PE_MAX_HEIGHT, PE_MAX_WIDTH)
    print(f"Setting up decoder with max lmx sequence length {MAX_LMX_SEQ_LEN}, vocab file {LMX_VOCAB_PATH}")
    decoder = OMRDecoder(MAX_LMX_SEQ_LEN, LMX_VOCAB_PATH)
    if device == "cpu":
        pretrained_mae_state_dict = torch.load(PRETRAINED_MAE_STATE_DICT_PATH, map_location=torch.device("cpu"))
    else:
        pretrained_mae_state_dict = torch.load(PRETRAINED_MAE_STATE_DICT_PATH)
    print(f"Loaded pretrained mae state dict from {PRETRAINED_MAE_STATE_DICT_PATH}")
    print("Setting up ViTOMR model")
    vitomr = ViTOMR(encoder, pretrained_mae_state_dict, decoder)
    vitomr = vitomr.to(device)

    base_img_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        DynamicResize(PATCH_SIZE, MAX_IMG_SEQ_LEN, PE_MAX_HEIGHT, PE_MAX_WIDTH, False)
    ])

    # slightly stronger augmentation since this training stage should be way easier
    camera_augment = v2.RandomApply(transforms=[
        v2.GaussianBlur(kernel_size=15, sigma=1),
        v2.GaussianNoise(sigma=0.03),
        v2.RandomRotation(degrees=(-2, 2), interpolation=InterpolationMode.BILINEAR),
        v2.RandomPerspective(distortion_scale=0.1, p=1),
        v2.ColorJitter(brightness=0.3, saturation=0.2, contrast=0.2, hue=0),
    ], p=AUGMENTATION_P)

    grandstaff_camera_augment = v2.Compose([
        v2.RandomPerspective(distortion_scale=0.1, p=1),
        v2.ColorJitter(brightness=0.3, saturation=0.2, contrast=0.2, hue=0),
    ])

    olimpic_img_transform = v2.Compose([base_img_transform, camera_augment])

    base_lmx_transform = PrepareLMXSequence(decoder.tokens_to_idxs)

    grand_staff = GrandStaffLMXDataset(GRAND_STAFF_ROOT_DIR, "samples.train.txt", img_transform=base_img_transform, lmx_transform=base_lmx_transform)
    olimpic = OlimpicDataset(OLIMPIC_SYNTHETIC_ROOT_DIR, "samples.train.txt", img_transform=olimpic_img_transform, lmx_transform=base_lmx_transform)

    train_dataset = ConcatDataset([
        GrandStaffOMRTrainWrapper(grand_staff, AUGMENTATION_P, transform=grandstaff_camera_augment),
        olimpic,
    ])

    grand_staff_validate = GrandStaffLMXDataset(GRAND_STAFF_ROOT_DIR, "samples.dev.txt", img_transform=base_img_transform, lmx_transform=base_lmx_transform)
    olimpic_synthetic_validate = OlimpicDataset(OLIMPIC_SYNTHETIC_ROOT_DIR, "samples.dev.txt", img_transform=base_img_transform)
    olimpic_scanned_validate = OlimpicDataset(OLIMPIC_SCANNED_ROOT_DIR, "samples.dev.txt", img_transform=base_img_transform)

    validation_dataset = ConcatDataset([
        GrandStaffOMRTrainWrapper(grand_staff_validate),
        olimpic_synthetic_validate,
        olimpic_scanned_validate,
    ])

    omr_train(vitomr, train_dataset, validation_dataset)

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
