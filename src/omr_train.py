import torch
from torch import nn
from pathlib import Path
from models import FineTuneOMREncoder, OMRDecoder, ViTOMR, OMRLoss
from datasets import GrandStaffLMXDataset, GrandStaffOMRTrainWrapper, OlimpicDataset
from config import GRAND_STAFF_ROOT_DIR, OLIMPIC_SCANNED_ROOT_DIR, OLIMPIC_SYNTHETIC_ROOT_DIR, LMX_BOS_TOKEN, LMX_EOS_TOKEN
from utils import DynamicResize, cosine_anneal_with_warmup, save_training_stats, ragged_collate_fn
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.transforms import v2, InterpolationMode
from torch.amp import autocast
from pre_train import PATCH_SIZE, PE_MAX_HEIGHT, PE_MAX_WIDTH
from dataclasses import dataclass
import time

MODEL_DIR_PATH = Path("omr_train")
CHECKPOINTS_DIR_PATH = MODEL_DIR_PATH / "checkpoints"
STATS_DIR_PATH = MODEL_DIR_PATH / "stats"

PRETRAINED_MAE_STATE_DICT_PATH = "mae_pre_train/pretrained_mae.pth"
ENCODER_FINE_TUNE_DEPTH = 12
MAX_IMG_SEQ_LEN = 512 # for DynamicResize
MAX_LMX_SEQ_LEN = 1536 # in tokens, max lmx token sequence length to support
LMX_VOCAB_PATH = "lmx_vocab.txt"
NUM_DECODER_LAYERS = 20

AUGMENTATION_P = 0.2
NUM_WORKERS = 26

EPOCHS = 60
CHECKPOINT_FREQ = 5
FINE_TUNE_BASE_LR = 2e-5 # 0.2x BASE_LR
FINE_TUNE_DECAY_FACTOR = 0.9
BASE_LR = 1e-4
MIN_LR = 1e-6
ADAMW_BETAS = (0.9, 0.95)
ADAMW_WEIGHT_DECAY = 1e-3
WARMUP_EPOCHS = 25 # step scheduler per-batch since doing so little epochs
BATCH_SIZE = 16 
GRAD_ACCUMULATION_STEPS = 4

ENCODER_DROPOUT = 0.025
TRANSITION_HEAD_DROPOUT = 0.025
DECODER_DROPOUT = 0.05
LABEL_SMOOTHING = 0.02

GRAD_SNAPSHOT_FREQ = 2 # within each epoch, how often (in optimizer steps) to snapshot gradient norms

# TODO:
# implement omr autoregressive beam search inference

class PrepareLMXSequence(nn.Module):
    def __init__(self, tokens_to_idxs):
        super().__init__()
        self.tokens_to_idxs = tokens_to_idxs

    def forward(self, lmx: str):
        tokens = [LMX_BOS_TOKEN]
        tokens += lmx.strip().split()
        tokens.append(LMX_EOS_TOKEN)
        return torch.tensor([self.tokens_to_idxs[token] for token in tokens])

def save_omr_training_state(path, vitomr, optimizer, scheduler):
    print(f"Saving omr training state to {path}")
    torch.save({
        "vitomr_state_dict": vitomr.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, path)

@dataclass
class GradNormSnapshot:
    optim_step_num: int
    encoder_norm: float 
    transition_head_norm: float 
    decoder_norm: float

    def calc_overall_step_num(self, epoch, num_batches_per_epoch):
        self.step_num = epoch * num_batches_per_epoch + self.optim_step_num
        return self

def calc_grad_norms(vitomr):
    grad_norms = []
    components = [vitomr.encoder, vitomr.transition_head, vitomr.decoder]
    for component in components:
        for param in component.parameters():
            component_norm = 0
            if param.grad is not None:
                component_norm += param.grad.data.norm(2).item()
            grad_norms.append(component_norm)
    return grad_norms

def snapshot_grad_norms(vitomr, optim_step_count):
    grad_norms = calc_grad_norms(vitomr)
    return GradNormSnapshot(optim_step_count, grad_norms[0], grad_norms[1], grad_norms[2])

def train_loop(vitomr, dataloader, loss_fn, optimizer, grad_accumulation_steps, scheduler, device, grad_snapshot_freq):
    print("Starting training")
    vitomr.train()
    epoch_loss = 0
    num_batches = len(dataloader)
    len_dataset = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    optim_step_count = 0 # with gradient accumulation, batch_idx != the actual amount of optimizer steps we've done. Track the latter to know when to snapshot gradients
    grad_norm_snapshots = []

    for batch_idx, batch in enumerate(dataloader):
        batch = [(x.to(device, non_blocking=True), y.to(device, non_blocking=True)) for x, y in batch]
        with autocast(device_type=device, dtype=torch.bfloat16):
            pred, target_seqs = vitomr(batch)
            loss = loss_fn(pred, target_seqs)
        epoch_loss += loss.item()
        loss.backward()

        if batch_idx % 100 == 0:
            current_ex = batch_idx * batch_size + len(batch)
            print(f"[{current_ex:>6d}/{len_dataset:>6d}]")

        if (batch_idx + 1) % grad_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            optimizer.step()
            optim_step_count += 1
            if optim_step_count % grad_snapshot_freq == 0:
                grad_norm_snapshots.append(snapshot_grad_norms(vitomr, optim_step_count))
            optimizer.zero_grad()
            scheduler.step()
    
    avg_loss = epoch_loss / num_batches
    print(f"Average training loss over this epoch: {avg_loss}")
    return avg_loss, grad_norm_snapshots

def validation_loop(vitomr, dataloader, loss_fn, device):
    print("Starting validation")
    vitomr.eval()
    epoch_loss = 0
    num_batches = len(dataloader)
    len_dataset = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    with torch.no_grad():
        with autocast(device_type=device, dtype=torch.bfloat16):
            for batch_idx, batch in enumerate(dataloader):
                batch = [(x.to(device, non_blocking=True), y.to(device, non_blocking=True)) for x, y in batch]
                pred, target_seqs = vitomr(batch)
                epoch_loss += loss_fn(pred, target_seqs).item()

                if batch_idx % 50 == 0:
                    current_ex = batch_idx * batch_size + len(batch)
                    print(f"[{current_ex:>5d}/{len_dataset:>5d}]")
            
    avg_loss = epoch_loss / num_batches
    print(f"Average validation loss for this epoch: {avg_loss}")
    return avg_loss

def omr_train(vitomr, train_dataset, validation_dataset, device):
    print("Model architecture\n--------------------")
    print(vitomr)
    encoder_params_count = sum(p.numel() for p in vitomr.encoder.parameters() if p.requires_grad)
    transition_head_params_count = sum(p.numel() for p in vitomr.transition_head.parameters() if p.requires_grad)
    decoder_params_count = sum(p.numel() for p in vitomr.decoder.parameters() if p.requires_grad)
    print(f"Trainable parameters count\n--------------------\nEncoder: {encoder_params_count}\nTransition head: {transition_head_params_count}\nDecoder: {decoder_params_count}\nTotal: {encoder_params_count + transition_head_params_count + decoder_params_count}") 

    print(f"Setting up DataLoaders with batch size {BATCH_SIZE}, shuffle, {NUM_WORKERS} workers, ragged collate function, pinned memory")
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=ragged_collate_fn, pin_memory=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=ragged_collate_fn, pin_memory=True)
    print(f"Dataset augmentation probability: {AUGMENTATION_P}")

    print(f"Creating optimizer parameter groups using base lr {BASE_LR} for transition head and decoder, {FINE_TUNE_BASE_LR} as encoder fine-tune base lr with {FINE_TUNE_DECAY_FACTOR} layer-wise decay factor")
    param_groups, layer_lrs = vitomr.create_fine_tune_param_groups(BASE_LR, FINE_TUNE_BASE_LR, FINE_TUNE_DECAY_FACTOR)
    print(f"Encoder fine-tune base lrs by layer: {layer_lrs}")

    print(f"Setting up AdamW with betas {ADAMW_BETAS}, weight decay {ADAMW_WEIGHT_DECAY}")
    optimizer = torch.optim.AdamW(param_groups, betas=ADAMW_BETAS, weight_decay=ADAMW_WEIGHT_DECAY)

    print(f"Accumulating gradients for {GRAD_ACCUMULATION_STEPS} steps for an effective batch size of {GRAD_ACCUMULATION_STEPS * BATCH_SIZE}")
    num_effective_batches = -(len(train_dataloader) // -GRAD_ACCUMULATION_STEPS)
    print(f"Setting up scheduler with {WARMUP_EPOCHS} warm-up epochs, {EPOCHS} total epochs, {MIN_LR} minimum learning rate, {num_effective_batches} (effective) batches per epoch")
    scheduler = cosine_anneal_with_warmup(optimizer, WARMUP_EPOCHS, EPOCHS, MIN_LR, num_train_batches=num_effective_batches)

    print(f"Snapshotting gradient norms every {GRAD_SNAPSHOT_FREQ} steps within each epoch")
    
    loss_fn = OMRLoss(vitomr.decoder.padding_idx, label_smoothing=LABEL_SMOOTHING)
    print(f"Using label smoothing of {loss_fn.label_smoothing} for cross entropy loss")

    MODEL_DIR_PATH.mkdir(exist_ok=True)
    CHECKPOINTS_DIR_PATH.mkdir(exist_ok=True)
    STATS_DIR_PATH.mkdir(exist_ok=True)
    print(f"Created directories {MODEL_DIR_PATH}, {CHECKPOINTS_DIR_PATH}, {STATS_DIR_PATH}")

    epoch_training_losses = []
    epoch_validation_losses = []
    epoch_lrs = [] # tuples of (base_lr, fine_tune_lr)
    epoch_grad_norm_snapshots = []

    print(f"OMR training for {EPOCHS} epochs. Checkpointing every {CHECKPOINT_FREQ} epochs")
    for i in range(EPOCHS):
        print(f"Epoch {i + 1}\n--------------------")
        base_lr = optimizer.param_groups[0]["lr"] # assuming transition head/decoder lr is the same
        fine_tune_base_lr = optimizer.param_groups[2]["lr"] # record the highest fine-tune lr all the decayed ones are based on
        print(f"Base learning rate at epoch start: {base_lr:>0.8f}\nFine-tune learning rate at epoch start: {fine_tune_base_lr:>0.8f}")
        epoch_lrs.append((base_lr, fine_tune_base_lr))

        train_start_time = time.perf_counter()
        epoch_train_loss, grad_norm_snapshots = train_loop(vitomr, train_dataloader, loss_fn, optimizer, GRAD_ACCUMULATION_STEPS, scheduler, device, GRAD_SNAPSHOT_FREQ)
        train_end_time = time.perf_counter()
        epoch_training_losses.append(epoch_train_loss)
        epoch_grad_norm_snapshots += [snapshot.calc_overall_step_num(i, num_effective_batches) for snapshot in grad_norm_snapshots]
        time_delta = train_end_time - train_start_time
        print(f"Time for this training epoch: {time_delta:>0.2f} seconds ({time_delta / 60:>0.2f} minutes)")

        epoch_validation_loss = validation_loop(vitomr, validation_dataloader, loss_fn, device)
        epoch_validation_losses.append(epoch_validation_loss)

        if (i + 1) % CHECKPOINT_FREQ == 0:
            print("Checkpointing model, optimizer, scheduler state dicts")
            checkpoint_path = CHECKPOINTS_DIR_PATH / f"epoch_{i+1}_checkpoint.pth"
            save_omr_training_state(checkpoint_path, vitomr, optimizer, scheduler)
            print("Checkpointing stats plots")
            save_training_stats(STATS_DIR_PATH, epoch_training_losses, epoch_validation_losses, epoch_lrs, epoch_grad_norm_snapshots, fine_tuning=True)

    print("Plotting final stats")
    save_training_stats(STATS_DIR_PATH, epoch_training_losses, epoch_validation_losses, epoch_lrs, epoch_grad_norm_snapshots, fine_tuning=True)
    print("Saving final omr training state")
    omr_train_state_path = MODEL_DIR_PATH / f"ending_omr_train_state.pth"
    save_omr_training_state(omr_train_state_path, vitomr, optimizer, scheduler)
    model_path = MODEL_DIR_PATH / "vitomr.pth"
    print(f"Saving final model state dict separately to {model_path}")
    torch.save(vitomr.state_dict(), model_path)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using device {device}")

print(f"Setting up encoder with patch size {PATCH_SIZE}, pe grid of {PE_MAX_HEIGHT} x {PE_MAX_WIDTH}, fine-tuning last {ENCODER_FINE_TUNE_DEPTH} layers")
encoder = FineTuneOMREncoder(PATCH_SIZE, PE_MAX_HEIGHT, PE_MAX_WIDTH, ENCODER_FINE_TUNE_DEPTH, transformer_dropout=ENCODER_DROPOUT)
print(f"Setting up decoder with max lmx sequence length {MAX_LMX_SEQ_LEN}, vocab file {LMX_VOCAB_PATH}")
decoder = OMRDecoder(MAX_LMX_SEQ_LEN, LMX_VOCAB_PATH, num_layers=NUM_DECODER_LAYERS, transformer_dropout=DECODER_DROPOUT)
if device == "cpu":
    pretrained_mae_state_dict = torch.load(PRETRAINED_MAE_STATE_DICT_PATH, map_location=torch.device("cpu"))
else:
    pretrained_mae_state_dict = torch.load(PRETRAINED_MAE_STATE_DICT_PATH)
print(f"Loaded pretrained mae state dict from {PRETRAINED_MAE_STATE_DICT_PATH}")
print("Setting up ViTOMR model")
vitomr = ViTOMR(encoder, pretrained_mae_state_dict, decoder, transition_head_dropout_p=TRANSITION_HEAD_DROPOUT)
vitomr = vitomr.to(device)

base_img_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    DynamicResize(PATCH_SIZE, MAX_IMG_SEQ_LEN, PE_MAX_HEIGHT, PE_MAX_WIDTH, False)
])

base_lmx_transform = PrepareLMXSequence(decoder.tokens_to_idxs)

if __name__ == "__main__":

    # slightly stronger augmentation since this training stage should be aided by the pre-training
    camera_augment = v2.RandomApply(transforms=[
        v2.GaussianBlur(kernel_size=15, sigma=1),
        v2.GaussianNoise(sigma=0.03),
        v2.RandomRotation(degrees=(-2, 2), interpolation=InterpolationMode.BILINEAR),
        v2.RandomPerspective(distortion_scale=0.15, p=1),
        v2.ColorJitter(brightness=0.3, saturation=0.2, contrast=0.2, hue=0),
    ], p=AUGMENTATION_P)

    grandstaff_camera_augment = v2.Compose([
        v2.RandomPerspective(distortion_scale=0.15, p=1),
        v2.ColorJitter(brightness=0.3, saturation=0.2, contrast=0.2, hue=0),
    ])

    olimpic_img_transform = v2.Compose([base_img_transform, camera_augment])

    grand_staff = GrandStaffLMXDataset(GRAND_STAFF_ROOT_DIR, "samples.train.txt", img_transform=base_img_transform, lmx_transform=base_lmx_transform)
    olimpic = OlimpicDataset(OLIMPIC_SYNTHETIC_ROOT_DIR, "samples.train.txt", img_transform=olimpic_img_transform, lmx_transform=base_lmx_transform)

    train_dataset = ConcatDataset([
        GrandStaffOMRTrainWrapper(grand_staff, AUGMENTATION_P, transform=grandstaff_camera_augment),
        olimpic,
    ])

    grand_staff_validate = GrandStaffLMXDataset(GRAND_STAFF_ROOT_DIR, "samples.dev.txt", img_transform=base_img_transform, lmx_transform=base_lmx_transform)
    olimpic_synthetic_validate = OlimpicDataset(OLIMPIC_SYNTHETIC_ROOT_DIR, "samples.dev.txt", img_transform=base_img_transform, lmx_transform=base_lmx_transform)
    olimpic_scanned_validate = OlimpicDataset(OLIMPIC_SCANNED_ROOT_DIR, "samples.dev.txt", img_transform=base_img_transform, lmx_transform=base_lmx_transform)

    validation_dataset = ConcatDataset([
        GrandStaffOMRTrainWrapper(grand_staff_validate),
        olimpic_synthetic_validate,
        olimpic_scanned_validate,
    ])

    omr_train(vitomr, train_dataset, validation_dataset, device)
