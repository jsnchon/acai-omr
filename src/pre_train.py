import torch
from datasets import GrandStaffLMXDataset, PreparedDataset, OlimpicDataset, GrandStaffPreTrainWrapper, OlimpicPreTrainWrapper, PreTrainWrapper
from utils import GRAND_STAFF_ROOT_DIR, PRIMUS_PREPARED_ROOT_DIR, DOREMI_PREPARED_ROOT_DIR, OLIMPIC_SYNTHETIC_ROOT_DIR, OLIMPIC_SCANNED_ROOT_DIR
from utils import DynamicResize, cosine_anneal_with_warmup, save_training_stats
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.transforms import InterpolationMode, v2
from models import MAE, MAELoss
from pathlib import Path
import time

MODEL_DIR_PATH = Path("mae_pre_train")
CHECKPOINTS_DIR_PATH = MODEL_DIR_PATH / "checkpoints"
STATS_DIR_PATH = MODEL_DIR_PATH / "stats"

# MAE constants
PATCH_SIZE = 16 # actual patch will be PATCH_SIZE x PATCH_SIZE
MASK_RATIO = 0.75
MAX_SEQ_LEN = 512 # max amount of tokens to allow images to be resized to
# in patches, positional embed dimensions. Larger images during pre-training will be center-cropped to fit. This can 
# be interpolated during inference after pre-training
PE_MAX_HEIGHT = 60 
PE_MAX_WIDTH = 200

# data constants
AUGMENTATION_P = 0.2 # probability to apply camera augmentation 
NUM_WORKERS = 24

# training hyperparameters
EPOCHS = 500
CHECKPOINT_FREQ = 50
BASE_LR = 1.5e-4 # max lr, used as annealing phase start and in warm-up phase lambda calculation
MIN_LR = 1e-6 # min lr of annealing phase 
ADAMW_BETAS = (0.9, 0.95) # lr/AdamW settings basically copied from the paper
ADAMW_WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 50 
BATCH_SIZE = 64

# collate ragged batch into a list of (input, target) tensors for the MAE logic to handle
def ragged_collate_fn(batch):
    collated_batch = []
    for example in batch:
        collated_batch.append((example[0], example[1]))
    return collated_batch

def save_pretraining_state(path, mae, optimizer, scheduler):
    print(f"Saving pretraining state to {path}")
    torch.save({
        "mae_state_dict": mae.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict()
    }, path)

def train_loop(mae, dataloader, loss_fn, optimizer, scheduler, device):
    print("Starting training")
    mae.train()
    epoch_loss = 0
    num_batches = len(dataloader)
    len_dataset = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    for batch_idx, batch in enumerate(dataloader):
        # move batch to right device
        batch = [(x.to(device, non_blocking=True), y.to(device, non_blocking=True)) for x, y in batch]
        pred, loss_mask, target = mae(batch)
        loss = loss_fn(pred, loss_mask, target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % 100 == 0:
            current_ex = batch_idx * batch_size + len(batch)
            print(f"[{current_ex:>6d}/{len_dataset:>6d}]")
    
    scheduler.step()
    avg_loss = epoch_loss / num_batches
    print(f"Average training loss over this epoch: {avg_loss}")
    return avg_loss

def validation_loop(mae, dataloader, loss_fn, device):
    print("Starting validation")
    mae.eval()
    epoch_loss = 0
    num_batches = len(dataloader)
    len_dataset = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = [(x.to(device, non_blocking=True), y.to(device, non_blocking=True)) for x, y in batch]
            pred, loss_mask, target = mae(batch)
            epoch_loss += loss_fn(pred, loss_mask, target).item()

            if batch_idx % 50 == 0:
                current_ex = batch_idx * batch_size + len(batch)
                print(f"[{current_ex:>5d}/{len_dataset:>5d}]")
        
    avg_loss = epoch_loss / num_batches
    print(f"Average validation loss for this epoch: {avg_loss}")
    return avg_loss

def pre_train(mae, train_dataset, validation_dataset):
    print("Model architecture\n--------------------")
    print(mae)
    params_count = sum(p.numel() for p in mae.parameters() if p.requires_grad)
    print(f"Trainable parameters count: {params_count}") 
    print(f"Setting up DataLoaders with batch size {BATCH_SIZE}, shuffle, {NUM_WORKERS} workers, pre train collate function, pinned memory")
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=ragged_collate_fn, pin_memory=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=ragged_collate_fn, pin_memory=True)
    print(f"Dataset augmentation probability: {AUGMENTATION_P}")
    print(f"Setting up AdamW with base learning rate {BASE_LR}, betas {ADAMW_BETAS}, weight decay {ADAMW_WEIGHT_DECAY}")
    optimizer = torch.optim.AdamW(mae.parameters(), lr=BASE_LR, betas=ADAMW_BETAS, weight_decay=ADAMW_WEIGHT_DECAY)
    print(f"Setting up scheduler with {WARMUP_EPOCHS} warm-up epochs, {EPOCHS} total epochs, {MIN_LR} minimum learning rate")
    scheduler = cosine_anneal_with_warmup(optimizer, WARMUP_EPOCHS, EPOCHS, MIN_LR)
    loss_fn = MAELoss()

    MODEL_DIR_PATH.mkdir(exist_ok=True)
    CHECKPOINTS_DIR_PATH.mkdir(exist_ok=True)
    STATS_DIR_PATH.mkdir(exist_ok=True)
    print(f"Created directories {MODEL_DIR_PATH}, {CHECKPOINTS_DIR_PATH}, {STATS_DIR_PATH}")

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using device {device}")

    mae = mae.to(device)
    epoch_training_losses = []
    epoch_validation_losses = []
    epoch_lrs = []

    print(f"Pretraining for {EPOCHS} epochs. Checkpointing every {CHECKPOINT_FREQ} epochs")
    for i in range(EPOCHS):
        print(f"Epoch {i + 1}\n--------------------")
        epoch_start_lr = optimizer.param_groups[0]["lr"]
        print(f"Learning rate at epoch start: {epoch_start_lr:>0.8f}")
        epoch_lrs.append(epoch_start_lr)

        train_start_time = time.perf_counter()
        epoch_train_loss = train_loop(mae, train_dataloader, loss_fn, optimizer, scheduler, device)
        train_end_time = time.perf_counter()
        epoch_training_losses.append(epoch_train_loss)
        time_delta = train_end_time - train_start_time
        print(f"Time for this training epoch: {time_delta:>0.2f} seconds ({time_delta / 60:>0.2f} minutes)")

        epoch_validation_loss = validation_loop(mae, validation_dataloader, loss_fn, device)
        epoch_validation_losses.append(epoch_validation_loss)

        if (i + 1) % CHECKPOINT_FREQ == 0:
            print("Checkpointing model, optimizer, scheduler state dicts")
            checkpoint_path = CHECKPOINTS_DIR_PATH / f"epoch_{i+1}_checkpoint.pth"
            save_pretraining_state(checkpoint_path, mae, optimizer, scheduler)
            print("Checkpointing stats plots")
            save_training_stats(STATS_DIR_PATH, epoch_training_losses, epoch_validation_losses, epoch_lrs)

    print("Plotting final stats")
    save_training_stats(STATS_DIR_PATH, epoch_training_losses, epoch_validation_losses, epoch_lrs)
    print("Saving final pretraining state")
    pretrain_state_path = MODEL_DIR_PATH / f"ending_pretrain_state.pth"
    save_pretraining_state(pretrain_state_path, mae, optimizer, scheduler)
    model_path = MODEL_DIR_PATH / "pretrained_mae.pth"
    print(f"Saving final model state dict separately to {model_path}")
    torch.save(mae.state_dict(), model_path)

mae = MAE(MASK_RATIO, PATCH_SIZE, PE_MAX_HEIGHT, PE_MAX_WIDTH)

# base transform for all images: convert to tensor, scale to patch divisible size within token budget
base_transform = v2.Compose([
    v2.ToImage(), # ToTensor is deprecated
    v2.ToDtype(torch.float32, scale=True),
    DynamicResize(PATCH_SIZE, MAX_SEQ_LEN, PE_MAX_HEIGHT, PE_MAX_WIDTH, True),
])

if __name__ == "__main__":
    print(f"MAE set up with mask ratio {MASK_RATIO} and patch size {PATCH_SIZE}")
    # base train datasets
    grand_staff = GrandStaffLMXDataset(GRAND_STAFF_ROOT_DIR, "samples.train.txt", img_transform=base_transform)
    primus = PreparedDataset(PRIMUS_PREPARED_ROOT_DIR, transform=base_transform)
    doremi = PreparedDataset(DOREMI_PREPARED_ROOT_DIR, transform=base_transform)
    olimpic = OlimpicDataset(OLIMPIC_SYNTHETIC_ROOT_DIR, "samples.train.txt", img_transform=base_transform)

    # augmentation to make image look like it was taken with a phone camera with AUGMENTATION_P probability
    camera_augment = v2.RandomApply(transforms=[
        v2.GaussianBlur(kernel_size=15, sigma=1),
        v2.GaussianNoise(sigma=0.03),
        v2.RandomRotation(degrees=(-1, 1), interpolation=InterpolationMode.BILINEAR),
        v2.RandomPerspective(distortion_scale=0.06, p=1),
        v2.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.2, hue=0),
    ], p=AUGMENTATION_P)

    # grandstaff specific camera augmentation (since dataset already has partially-augmented variants)
    grandstaff_camera_augment = v2.Compose([
        v2.RandomPerspective(distortion_scale=0.08, p=1),
        v2.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.2, hue=0),
    ])

    # initialize concatenated pre-train dataset of all the wrappers
    train_dataset = ConcatDataset([
        PreTrainWrapper(primus, transform=camera_augment),
        PreTrainWrapper(doremi, transform=camera_augment),
        GrandStaffPreTrainWrapper(grand_staff, augment_p=AUGMENTATION_P, transform=grandstaff_camera_augment),
        OlimpicPreTrainWrapper(olimpic, transform=camera_augment),
    ])

    # validation dataset setup
    grand_staff_validate = GrandStaffLMXDataset(GRAND_STAFF_ROOT_DIR, "samples.dev.txt", img_transform=base_transform)
    olimpic_synthetic_validate = OlimpicDataset(OLIMPIC_SYNTHETIC_ROOT_DIR, "samples.dev.txt", img_transform=base_transform)
    olimpic_scanned_validate = OlimpicDataset(OLIMPIC_SCANNED_ROOT_DIR, "samples.dev.txt", img_transform=base_transform)

    validation_dataset = ConcatDataset([
        GrandStaffPreTrainWrapper(grand_staff_validate),
        OlimpicPreTrainWrapper(olimpic_synthetic_validate),
        OlimpicPreTrainWrapper(olimpic_scanned_validate),
    ])

    pre_train(mae, train_dataset, validation_dataset)
