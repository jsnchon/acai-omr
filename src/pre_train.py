import torch
from torch import nn
from datasets import GrandStaffLMXDataset, PreparedDataset, OlimpicDataset, GrandStaffPreTrainWrapper, OlimpicPreTrainWrapper, PreTrainWrapper
from utils import GRAND_STAFF_ROOT_DIR, PRIMUS_PREPARED_ROOT_DIR, DOREMI_PREPARED_ROOT_DIR, OLIMPIC_SYNTHETIC_ROOT_DIR
from torch.utils.data import ConcatDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2

# musical scores have very small objects that convey important information, so maintaining resolution is very
# important. Instead of down/upsampling images to get them to fit ViT structure, resize them to a similar size
class PatchDivisibleResize(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    # img should be a PIL image or a tensor of shape C x H x W
    def forward(self, img):
        if torch.is_tensor(img):
            _, h, w = img.shape
        else:
            w, h = img.size

        # resize dimensions to nearest (lower) size evenly divisible by patch_size 
        # floor division could be 0, so take max with self.patch_size to enforce minimum dim of patch_size
        new_w = max(w // self.patch_size * self.patch_size, self.patch_size)
        new_h = max(h // self.patch_size * self.patch_size, self.patch_size)
        resize = v2.Resize(
            size=(new_h, new_w),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        return resize(img)

PATCH_SIZE = 16 # actual patch will be PATCH_SIZE x PATCH_SIZE
AUGMENTATION_P = 0.5 # probability to apply camera augmentation with
# base transform for all images: scale to nearest resolution divisible by patch size
base_transform = v2.Compose([
    PatchDivisibleResize(PATCH_SIZE),
    v2.ToImage(), # ToTensor is deprecated
    v2.ToDtype(torch.float32, scale=True),
])

# base datasets
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

# initialize concated pre-train dataset of all the wrappers
pre_train_dataset = ConcatDataset([
    PreTrainWrapper(primus, transform=camera_augment),
    PreTrainWrapper(doremi, transform=camera_augment),
    GrandStaffPreTrainWrapper(grand_staff, augment_p=AUGMENTATION_P, transform=grandstaff_camera_augment),
    OlimpicPreTrainWrapper(olimpic, transform=camera_augment),
])


from utils import sample_pre_train_dataset
sample_pre_train_dataset(pre_train_dataset, 15, PATCH_SIZE)
# so specify base transform on datasets to crop images
# specify camera augment for everything except grandstaff, augment to get to camera
# on distorted images from grandstaff