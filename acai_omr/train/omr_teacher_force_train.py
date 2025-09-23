import torch
from torch import nn
from pathlib import Path
from acai_omr.models.models import FineTuneOMREncoder, OMRDecoder, ScheduledSamplingViTOMR, OMRCELoss
from acai_omr.train.datasets import GrandStaffLMXDataset, GrandStaffOMRTrainWrapper, OlimpicDataset
from acai_omr.config import GRAND_STAFF_ROOT_DIR, OLIMPIC_SCANNED_ROOT_DIR, OLIMPIC_SYNTHETIC_ROOT_DIR, LMX_BOS_TOKEN, LMX_EOS_TOKEN
from acai_omr.utils.utils import DynamicResize, cosine_anneal_with_warmup, ragged_collate_fn, StepCounter
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2, InterpolationMode
from torch.amp import autocast
from acai_omr.train.pre_train import PATCH_SIZE, PE_MAX_HEIGHT, PE_MAX_WIDTH
import time
import pandas as pd
from dataclasses import dataclass

LOG_DIR = "runs/tf_omr_train"

MODEL_DIR_PATH = Path("tf_omr_train")
CHECKPOINTS_DIR_PATH = MODEL_DIR_PATH / "checkpoints"

PRETRAINED_MAE_STATE_DICT_PATH = "mae_pre_train/pretrained_mae.pth"
ENCODER_FINE_TUNE_DEPTH = 12
MAX_IMG_SEQ_LEN = 512 # for DynamicResize
MAX_LMX_SEQ_LEN = 1536 # in tokens, max lmx token sequence length to support
LMX_VOCAB_PATH = "lmx_vocab.txt"
NUM_DECODER_LAYERS = 12

# training settings
EPOCHS = 50
CHECKPOINT_FREQ = 10
FINE_TUNE_BASE_LR = 1e-5 # 0.1x base lr
FINE_TUNE_DECAY_FACTOR = 0.9
BASE_LR = 1e-4
MIN_LR = 1e-6
ADAMW_BETAS = (0.9, 0.95)
ADAMW_WEIGHT_DECAY = 0.01
WARMUP_EPOCHS = 3 # step scheduler per-batch since doing so little epochs
BATCH_SIZE = 16
GRAD_ACCUMULATION_STEPS = 4
NUM_WORKERS = 26

# regularization settings
AUGMENTATION_P = 0.4
ENCODER_DROPOUT = 0.05
TRANSITION_HEAD_DROPOUT = 0.05
DECODER_DROPOUT = 0.1
LABEL_SMOOTHING = 0.0

# teacher forcing/scheduled sampling settings. Teacher forcing prob decreases linearly, tau decreases exponentially
INITIAL_TEACHER_FORCING_PROB = 1.0
MIN_TEACHER_FORCING_PROB = 0.1
INITIAL_TAU = 5.0
MIN_TAU = 0.1
TF_ANNEAL_EPOCHS = 40 # number of epochs to anneal tf prob and tau down to. Remaining epochs, they'll remain at their floor
SOFT_EPOCHS = EPOCHS // 2

@dataclass
class TFConfig:
    tf_prob: float
    tau: float
    use_hard_sampling: bool

class TFScheduler:
    def __init__(self, tf_config: TFConfig, init_tf_prob, min_tf_prob, init_tau, min_tau, soft_epochs, anneal_epochs, num_steps_per_epoch):
        self.tf_config = tf_config
        self.init_tf_prob = init_tf_prob
        self.min_tf_prob = min_tf_prob
        self.init_tau = init_tau
        self.min_tau = min_tau
        self.soft_steps = soft_epochs * num_steps_per_epoch
        self.anneal_steps = anneal_epochs * num_steps_per_epoch
        self.step_count = 0

    def step(self):
        if self.step_count >= self.soft_steps:
            self.tf_config.use_hard_sampling = True

        anneal_progress = self.step_count / self.anneal_steps
        self.tf_config.tf_prob = max(self.init_tf_prob - (self.init_tf_prob - self.min_tf_prob) * anneal_progress, self.min_tf_prob)
        self.tf_config.tau = max(self.init_tau * (self.min_tau / self.init_tau) ** anneal_progress, self.min_tau)

        self.step_count += 1

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

def train_loop(vitomr: ScheduledSamplingViTOMR, dataloader, loss_fn, optimizer, scheduler, device, grad_accumulation_steps, tf_config: TFConfig, tf_scheduler: TFScheduler, writer, counter):
    print("Starting training")
    vitomr.train()
    epoch_loss = 0
    accumulated_losses = []
    num_batches = len(dataloader)
    len_dataset = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    for batch_idx, batch in enumerate(dataloader):
        batch = [(x.to(device, non_blocking=True), y.to(device, non_blocking=True)) for x, y in batch]
        with autocast(device_type=device, dtype=torch.bfloat16):
            pred, target_seqs = vitomr.forward_train(batch, tf_config.tf_prob, tf_config.tau, tf_config.use_hard_sampling)
            loss = loss_fn(pred, target_seqs)
        epoch_loss += loss.item()
        accumulated_losses.append(loss.item())
        loss.backward()

        if batch_idx % 100 == 0:
            current_ex = batch_idx * batch_size + len(batch)
            print(f"[{current_ex:>5d}/{len_dataset:>5d}]")

        if (batch_idx + 1) % grad_accumulation_steps == 0 or (batch_idx + 1) == num_batches:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            tf_scheduler.step()

            writer.add_scalar(f"train/loss", sum(accumulated_losses) / len(accumulated_losses), counter.global_step)
            writer.add_scalar(f"train/hyperparams/base_lr", optimizer.param_groups[0]["lr"], counter.global_step)
            writer.add_scalar(f"train/hyperparams/fine_tune_base_lr", optimizer.param_groups[2]["lr"], counter.global_step)
            writer.add_scalar(f"train/hyperparams/teacher_forcing_prob", tf_config.tf_prob, counter.global_step)
            writer.add_scalar(f"train/hyperparams/tau", tf_config.tau, counter.global_step)
            accumulated_losses = []
            counter.increment()
    
    avg_loss = epoch_loss / num_batches
    print(f"Average training loss over this epoch: {avg_loss}")
    return avg_loss

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
                pred, target_seqs = vitomr.forward_eval(batch)
                epoch_loss += loss_fn(pred, target_seqs).item()

                if batch_idx % 50 == 0:
                    current_ex = batch_idx * batch_size + len(batch)
                    print(f"[{current_ex:>5d}/{len_dataset:>5d}]")
            
    avg_loss = epoch_loss / num_batches
    print(f"Average validation loss for this epoch: {avg_loss}")
    return avg_loss

# this should be called at each epoch's start (ie before the train loop). Both epoch and max_epochs should be 0-indexed
def calc_teacher_forcing_prob(epoch, tf_anneal_epochs, initial_prob, min_prob):
    if epoch < tf_anneal_epochs:
        progress = epoch / tf_anneal_epochs
        return initial_prob - (initial_prob - min_prob) * progress
    else:
        return min_prob

# same use as calc_teacher_forcing_prob
def calc_tau(epoch, tf_anneal_epochs, initial_tau, min_tau):
    if epoch < tf_anneal_epochs:
        progress = epoch / tf_anneal_epochs
        return initial_tau * (min_tau / initial_tau) ** progress
    else:
        return min_tau

def omr_teacher_force_train(vitomr, train_dataset, validation_dataset, device):
    MODEL_DIR_PATH.mkdir()
    CHECKPOINTS_DIR_PATH.mkdir()
    print(f"Created directories {MODEL_DIR_PATH}, {CHECKPOINTS_DIR_PATH}\n")

    print(f"Model architecture\n{'-' * 50}\n{vitomr}")
    encoder_params_count = sum(p.numel() for p in vitomr.encoder.parameters() if p.requires_grad)
    transition_head_params_count = sum(p.numel() for p in vitomr.transition_head.parameters() if p.requires_grad)
    decoder_params_count = sum(p.numel() for p in vitomr.decoder.parameters() if p.requires_grad)
    print(f"Trainable parameters count\nEncoder: {encoder_params_count}\nTransition head: {transition_head_params_count}\nDecoder: {decoder_params_count}\nTotal: {encoder_params_count + transition_head_params_count + decoder_params_count}\n") 

    print(f"General hyperparameters\n{'-' * 50}\nEpochs: {EPOCHS}\nWarmup epochs: {WARMUP_EPOCHS}\nCheckpoint frequency: {CHECKPOINT_FREQ} epochs\n" \
          f"Base lr: {BASE_LR}\nFine tune base lr: {FINE_TUNE_BASE_LR}, layer-wise decay factor of {FINE_TUNE_DECAY_FACTOR}\nMinimum lr: {MIN_LR}\n" \
          f"AdamW betas: {ADAMW_BETAS}, weight decay: {ADAMW_WEIGHT_DECAY}\nBatch size: {BATCH_SIZE}\nGradient accumulation steps: {GRAD_ACCUMULATION_STEPS}\n" \
          f"Number of DataLoader workers: {NUM_WORKERS}\n")

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=ragged_collate_fn, pin_memory=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=ragged_collate_fn, pin_memory=True)

    param_groups, layer_lrs = vitomr.create_fine_tune_param_groups(BASE_LR, FINE_TUNE_BASE_LR, FINE_TUNE_DECAY_FACTOR)
    print(f"Encoder fine-tune base lrs by layer: {layer_lrs}\n")
    print(f"Regularization hyperparameters\n{'-' * 50}\nImage augmentation probability: {AUGMENTATION_P}\nEncoder dropout: {ENCODER_DROPOUT}\n" \
          f"Transition head dropout: {TRANSITION_HEAD_DROPOUT}\nDecoder dropout: {DECODER_DROPOUT}\nLabel smoothing: {LABEL_SMOOTHING}\n")

    optimizer = torch.optim.AdamW(param_groups, betas=ADAMW_BETAS, weight_decay=ADAMW_WEIGHT_DECAY)

    num_batches = -(len(train_dataloader) // -GRAD_ACCUMULATION_STEPS)
    scheduler = cosine_anneal_with_warmup(optimizer, WARMUP_EPOCHS, EPOCHS, MIN_LR, num_train_batches=num_batches)
    
    loss_fn = OMRCELoss(vitomr.decoder.pad_idx, label_smoothing=LABEL_SMOOTHING)

    print(f"Teacher forcing hyperparameters\n{'-' * 50}\nInitial teacher forcing per-token probability: {INITIAL_TEACHER_FORCING_PROB}\n" \
          f"Minimum teacher forcing probability: {MIN_TEACHER_FORCING_PROB}\nInitial Gumbel-Softmax tau: {INITIAL_TAU}\n" \
          f"Minimum Gumbel-Softmax tau: {MIN_TAU}\nNumber of epochs with soft prediction sampling: {SOFT_EPOCHS}\n" \
          f"Annealing teacher forcing probability and tau over {TF_ANNEAL_EPOCHS} epochs")

    writer = SummaryWriter(LOG_DIR, max_queue=50)
    counter = StepCounter()
    tf_config = TFConfig(INITIAL_TEACHER_FORCING_PROB, INITIAL_TAU, False)
    tf_scheduler = TFScheduler(tf_config, INITIAL_TEACHER_FORCING_PROB, MIN_TEACHER_FORCING_PROB, INITIAL_TAU, MIN_TAU, SOFT_EPOCHS, TF_ANNEAL_EPOCHS, num_batches)
    epoch_stats_df = pd.DataFrame(columns=["train_loss", "validation_loss", "base_lr", "fine_tune_base_lr", "teacher_forcing_prob", "tau", "use_hard_sampling"])

    print(f"OMR training for {EPOCHS} epochs. Checkpointing every {CHECKPOINT_FREQ} epochs")
    for i in range(EPOCHS):
        print(f"\nEpoch {i + 1}\n{'-' * 50}")
        base_lr = optimizer.param_groups[0]["lr"] # assuming transition head/decoder lr is the same
        fine_tune_base_lr = optimizer.param_groups[2]["lr"] # record the highest fine-tune lr all the decayed ones are based on
        print(f"Hyperparameters at epoch start:")
        print(f"Base learning rate: {base_lr:>0.8f}\nFine-tune learning rate: {fine_tune_base_lr:>0.8f}")
        print(f"Teacher forcing probability: {tf_config.tf_prob:>0.8f}\nGumbel-softmax tau: {tf_config.tau:>0.8f}\nUsing hard sampling: {tf_config.use_hard_sampling}")

        train_start_time = time.perf_counter()
        epoch_train_loss = train_loop(vitomr, train_dataloader, loss_fn, optimizer, scheduler, device, GRAD_ACCUMULATION_STEPS, tf_config, tf_scheduler, writer, counter)
        train_end_time = time.perf_counter()
        time_delta = train_end_time - train_start_time
        print(f"Time for this training epoch: {time_delta:>0.2f} seconds ({time_delta / 60:>0.2f} minutes)")

        epoch_validation_loss = validation_loop(vitomr, validation_dataloader, loss_fn, device)

        writer.add_scalars("epoch", {"train_loss": epoch_train_loss, "validation_loss": epoch_validation_loss}, counter.global_step)
        epoch_stats = [epoch_train_loss, epoch_validation_loss, base_lr, fine_tune_base_lr, tf_config.tf_prob, tf_config.tau, tf_config.use_hard_sampling]
        epoch_stats_df.loc[i] = epoch_stats

        if (i + 1) % CHECKPOINT_FREQ == 0:
            print("Checkpointing model, optimizer, scheduler state dicts")
            checkpoint_path = CHECKPOINTS_DIR_PATH / f"epoch_{i+1}_checkpoint.pth"
            save_omr_training_state(checkpoint_path, vitomr, optimizer, scheduler)
            print("Saving training stats csv")
            epoch_stats_df.to_csv((MODEL_DIR_PATH / "training_stats.csv"))
            writer.flush()

    print("Saving final omr training state")
    omr_train_state_path = MODEL_DIR_PATH / "ending_omr_train_state.pth"
    save_omr_training_state(omr_train_state_path, vitomr, optimizer, scheduler)
    model_path = MODEL_DIR_PATH / "vitomr.pth"
    print(f"Saving final model state dict separately to {model_path}")
    torch.save(vitomr.state_dict(), model_path)
    
    epoch_stats_df.to_csv((MODEL_DIR_PATH / "training_stats.csv"))
    writer.flush()

# constructing/loading the model definition into memory can be intensive, so instead of doing this whenever this module is imported,
# separate it into a function that can be called when it's needed
def set_up_omr_teacher_force_train():
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
    print("Setting up ViTOMR model\n")
    vitomr = ScheduledSamplingViTOMR(encoder, pretrained_mae_state_dict, decoder, transition_head_dropout=TRANSITION_HEAD_DROPOUT)
    vitomr = vitomr.to(device)

    base_img_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        DynamicResize(PATCH_SIZE, MAX_IMG_SEQ_LEN, PE_MAX_HEIGHT, PE_MAX_WIDTH, False)
    ])

    base_lmx_transform = PrepareLMXSequence(decoder.tokens_to_idxs)

    return vitomr, base_img_transform, base_lmx_transform, device

if __name__ == "__main__":
    vitomr, base_img_transform, base_lmx_transform, device = set_up_omr_teacher_force_train()

    # slightly stronger augmentation since this training stage should be aided by the pre-training
    camera_augment = v2.RandomApply(transforms=[
        v2.GaussianBlur(kernel_size=15, sigma=(0.2, 0.7)),
        v2.GaussianNoise(sigma=0.03),
        v2.RandomRotation(degrees=(-2, 2), interpolation=InterpolationMode.BILINEAR),
        v2.RandomPerspective(distortion_scale=0.2, p=1),
        v2.ColorJitter(brightness=0.15, saturation=0.2, contrast=0.2, hue=0),
    ], p=AUGMENTATION_P)

    grandstaff_camera_augment = v2.Compose([
        v2.RandomPerspective(distortion_scale=0.2, p=1),
        v2.ColorJitter(brightness=0.15, saturation=0.2, contrast=0.2, hue=0),
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

    omr_teacher_force_train(vitomr, train_dataset, validation_dataset, device)
