import torch
from datasets import GrandStaffLMXDataset, PreparedDataset, OlimpicDataset, GrandStaffPreTrainWrapper, OlimpicPreTrainWrapper, PreTrainWrapper
from utils import GRAND_STAFF_ROOT_DIR, PRIMUS_PREPARED_ROOT_DIR, DOREMI_PREPARED_ROOT_DIR, OLIMPIC_SYNTHETIC_ROOT_DIR, OLIMPIC_SCANNED_ROOT_DIR
from utils import BucketBatchSampler, PatchDivisibleResize
import timm
from torch.utils.data import ConcatDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2
from torchvision.models.vision_transformer import VisionTransformer
import logging

logging.basicConfig(level=logging.DEBUG)

PATCH_SIZE = 16 # actual patch will be PATCH_SIZE x PATCH_SIZE
AUGMENTATION_P = 0.5 # probability to apply camera augmentation 

# instead of stacking image tensors (which isn't possible since they're likely of different shapes), 
# return a list of input img tensors, a list of target img tensors, and deal with the embedding/stacking 
# in the MAE forward method
def pre_train_collate_fn(batch):
    input_img_tensors = []
    target_img_tensors = []
    for example in batch:
        input_img_tensors.append(example[0])
        target_img_tensors.append(example[1])
    return input_img_tensors, target_img_tensors

if __name__ == "__main__":
    # base transform for all images: scale to nearest resolution divisible by patch size
    base_transform = v2.Compose([
        PatchDivisibleResize(PATCH_SIZE),
        v2.ToImage(), # ToTensor is deprecated
        v2.ToDtype(torch.float32, scale=True),
    ])

    # base train datasets
    grand_staff = GrandStaffLMXDataset(GRAND_STAFF_ROOT_DIR, "samples.train.txt", transform=base_transform)
    primus = PreparedDataset(PRIMUS_PREPARED_ROOT_DIR, transform=base_transform)
    doremi = PreparedDataset(DOREMI_PREPARED_ROOT_DIR, transform=base_transform)
    olimpic = OlimpicDataset(OLIMPIC_SYNTHETIC_ROOT_DIR, "samples.train.txt", transform=base_transform)

    # augmentation to make image look like it was taken with a phone camera with AUGMENTATION_P probability
    camera_augment = v2.RandomApply(transforms=[
        v2.GaussianBlur(kernel_size=15, sigma=1.2),
        v2.GaussianNoise(sigma=0.05),
        v2.RandomRotation(degrees=(-1, 1), interpolation=InterpolationMode.BILINEAR),
        v2.RandomPerspective(distortion_scale=0.08, p=1),
        v2.ColorJitter(brightness=0.3, saturation=0.2, contrast=0.2, hue=0.1),
    ], p=AUGMENTATION_P)

    # grandstaff specific camera augmentation (since dataset already has partially-augmented variants)
    grandstaff_camera_augment = v2.Compose([
        v2.RandomPerspective(distortion_scale=0.08, p=1),
        v2.ColorJitter(brightness=0.3, saturation=0.2, contrast=0.25, hue=0.1),
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

    encoder = timm.create_model("vit_large_patch16_224.orig_in21k", pretrained=True)
    # pretrained data transfomr: {'input_size': (3, 224, 224), 'interpolation': 'bicubic', 'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 'crop_pct': 0.875, 'crop_mode': 'center'}
    """Image size isn't enforced, and ours is variable so specify a dummy size. As MAE paper states, decoder can 
    be lightweight. Number of attention heads is same as ViT_L but only use 2 blocks and halve
    hidden and MLP dim"""
    decoder = VisionTransformer(image_size=128, patch_size=PATCH_SIZE, num_layers=2, num_heads=16, hidden_dim=512, mlp_dim=2048)
    #print(decoder)