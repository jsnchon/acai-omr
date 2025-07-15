import torch
from datasets import GrandStaffLMXDataset, PreparedDataset, OlimpicDataset, GrandStaffPreTrainWrapper, OlimpicPreTrainWrapper, PreTrainWrapper
from utils import GRAND_STAFF_ROOT_DIR, PRIMUS_PREPARED_ROOT_DIR, DOREMI_PREPARED_ROOT_DIR, OLIMPIC_SYNTHETIC_ROOT_DIR, OLIMPIC_SCANNED_ROOT_DIR
from utils import DynamicResize
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2
from models import MAE, MAELoss
import logging

# TODO: trace out lr with scheduler to make sure it looks good, write a test for it
# TODO: in utils create function that reshapes prediction into image and displays it using matplot
# TODO: ensure (with a little model surgery) can transfer state dicts between MAEEncoder and Encoder

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
BASE_LR = 0.01 # max lr, used as annealing phase start and in warm-up phase lambda calculation
MIN_LR = 1e-6 # min lr of annealing phase 
WARMUP_EPOCHS = 25 
BATCH_SIZE = 32

# collate ragged batch into a list of tensors for the MAE logic to handle
def pre_train_collate_fn(batch):
    input_img_tensors = []
    target_img_tensors = []
    for example in batch:
        input_img_tensors.append(example[0])
        target_img_tensors.append(example[1])
    return input_img_tensors, target_img_tensors

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

    validate_datest = ConcatDataset([
        GrandStaffPreTrainWrapper(grand_staff_validate),
        OlimpicPreTrainWrapper(olimpic_synthetic_validate),
        OlimpicPreTrainWrapper(olimpic_scanned_validate),
    ])

    logger.info(f"Setting up MAE with mask ratio {MASK_RATIO} and patch size {PATCH_SIZE}")
    mae = MAE(MASK_RATIO, PATCH_SIZE)
    for name, module in mae.named_modules():
        pass
        # logger.debug(name, module)
    logger.info("Compiling MAE model")
    compiled_mae = torch.compile(mae, fullgraph=True)
    # dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1, collate_fn=pre_train_collate_fn, pin_memory=True)

    # make sure to step scheduler at epoch start so lr starts at bottom of warmup instead of optim base lr