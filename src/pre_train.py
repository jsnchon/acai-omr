import torch
from datasets import GrandStaffLMXDataset, PreparedDataset, OlimpicDataset, GrandStaffPreTrainWrapper, OlimpicPreTrainWrapper, PreTrainWrapper
from utils import GRAND_STAFF_ROOT_DIR, PRIMUS_PREPARED_ROOT_DIR, DOREMI_PREPARED_ROOT_DIR, OLIMPIC_SYNTHETIC_ROOT_DIR, OLIMPIC_SCANNED_ROOT_DIR
from utils import DynamicResize, cosine_anneal_with_warmup
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2
from models import MAE, MAELoss
import logging
from pathlib import Path

# TODO: finish training loop logic
# TODO: in utils create function that reshapes prediction into image and displays it using matplot
# TODO: ensure (with a little model surgery) can transfer state dicts between MAEEncoder and Encoder

MODEL_DIR = "mae_pre_train"

# MAE constants
PATCH_SIZE = 16 # actual patch will be PATCH_SIZE x PATCH_SIZE
MASK_RATIO = 0.75
MAX_SEQ_LEN = 512 # max amount of tokens to allow images to be resized to

# data constants
AUGMENTATION_P = 0.3 # probability to apply camera augmentation 
NUM_WORKERS = 24

# training hyperparameters
EPOCHS = 500
CHECKPOINT_FREQ = 50
BASE_LR = 2e-4 # max lr, used as annealing phase start and in warm-up phase lambda calculation
MIN_LR = 1e-6 # min lr of annealing phase 
ADAMW_BETAS = (0.9, 0.95) # lr/AdamW settings basically copied from the paper
ADAMW_WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 25 
BATCH_SIZE = 32

# collate ragged batch into a list of (input, target) tensors for the MAE logic to handle
def pre_train_collate_fn(batch):
    collated_batch = []
    for example in batch:
        collated_batch.append((example[0], example[1]))
    return collated_batch

if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # base transform for all images: scale to nearest resolution divisible by patch size
    base_transform = v2.Compose([
        v2.ToImage(), # ToTensor is deprecated
        v2.ToDtype(torch.float32, scale=True),
        DynamicResize(PATCH_SIZE, MAX_SEQ_LEN),
    ])

    # base train datasets
    grand_staff = GrandStaffLMXDataset(GRAND_STAFF_ROOT_DIR, "samples.train.txt", transform=base_transform)
    primus = PreparedDataset(PRIMUS_PREPARED_ROOT_DIR, transform=base_transform)
    doremi = PreparedDataset(DOREMI_PREPARED_ROOT_DIR, transform=base_transform)
    olimpic = OlimpicDataset(OLIMPIC_SYNTHETIC_ROOT_DIR, "samples.train.txt", transform=base_transform)

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
    grand_staff_validate = GrandStaffLMXDataset(GRAND_STAFF_ROOT_DIR, "samples.dev.txt", transform=base_transform)
    olimpic_synthetic_validate = OlimpicDataset(OLIMPIC_SYNTHETIC_ROOT_DIR, "samples.dev.txt", transform=base_transform)
    olimpic_scanned_validate = OlimpicDataset(OLIMPIC_SCANNED_ROOT_DIR, "samples.dev.txt", transform=base_transform)

    validation_dataset = ConcatDataset([
        GrandStaffPreTrainWrapper(grand_staff_validate),
        OlimpicPreTrainWrapper(olimpic_synthetic_validate),
        OlimpicPreTrainWrapper(olimpic_scanned_validate),
    ])

    logger.info(f"Setting up MAE with mask ratio {MASK_RATIO} and patch size {PATCH_SIZE}")
    mae = MAE(MASK_RATIO, PATCH_SIZE)
    logger.info(f"Setting up DataLoaders with batch size {BATCH_SIZE}, shuffle, {NUM_WORKERS} workers, pre train collate function, pinned memory")
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=pre_train_collate_fn, pin_memory=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=pre_train_collate_fn, pin_memory=True)
    logger.info(f"Setting up AdamW with base learning rate {BASE_LR}, betas {ADAMW_BETAS}, weight decay {ADAMW_WEIGHT_DECAY}")
    optimizer = torch.optim.AdamW(mae.parameters(), lr=BASE_LR, betas=ADAMW_BETAS, weight_decay=ADAMW_WEIGHT_DECAY)
    logger.info(f"Setting up scheduler with {WARMUP_EPOCHS} warm-up epochs, {EPOCHS} total epochs, {MIN_LR} minimum learning rate")
    scheduler = cosine_anneal_with_warmup(optimizer, WARMUP_EPOCHS, EPOCHS, MIN_LR)
    loss_fn = MAELoss()

    model_dir_path = Path(MODEL_DIR)
    checkpoints_path = MODEL_DIR / "checkpoints"
    stats_dir = MODEL_DIR / "stats"
    model_dir_path.mkdir(exist_ok=True)
    checkpoints_path.mkdir(exist_ok=True)
    stats_dir.mkdir(exist_ok=True)
    logger.info(f"Created directories {model_dir_path}, {checkpoints_path}, {stats_dir}")

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    logger.info(f"Using device {device}")

    for i in range(EPOCHS):
        logger.info(f"Epoch {i + 1}\n--------------------")
        # train_loop(model, train_dataloader, loss_fn, optimizer, scheduler, device)
        # validation_loop(model, validation_dataloader, loss_fn, device)
        # checkpoint(i)