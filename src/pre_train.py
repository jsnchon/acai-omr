import torch
from torch import nn
from datasets import GrandStaffLMXDataset, PreparedDataset, OlimpicDataset, GrandStaffPreTrainWrapper, OlimpicPreTrainWrapper, PreTrainWrapper
from utils import GRAND_STAFF_ROOT_DIR, PRIMUS_PREPARED_ROOT_DIR, DOREMI_PREPARED_ROOT_DIR, OLIMPIC_SYNTHETIC_ROOT_DIR, OLIMPIC_SCANNED_ROOT_DIR
from torch.utils.data import ConcatDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2
from torchvision.models.vision_transformer import VisionTransformer
from torch.utils.data.sampler import BatchSampler
import numpy as np
import timm
import logging

logging.basicConfig(level=logging.DEBUG)

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

"""
DataLoader batching requires non-ragged batches, ie all examples are of the same length. A collate function to pad based on
the largest image in the batch will deal with this, but it's much more efficient if we ensure all batches have similarly sized
images to prevent over-padding. BucketBatchSampler takes a Dataset instance dataset and list of tuples bucket_boundaries and
groups examples within dataset together based on image size. Each tuple in bucket_boundaries specifies a max_h, max_w for a bucket. 
Images will be placed in the smallest bucket they can fit in. bucket_boundaries must be in ascending order of specified size

Note that this sampler buckets only based on the size of the first item returned by an indexing of dataset. The
assumption is this is always an image, and that any other images retrieved with it are of a similar size."""
class BucketBatchSampler(BatchSampler):
    def __init__(self, dataset, bucket_boundaries, batch_size, shuffle=True):
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"Initializing bucketed batch sampler for dataset {dataset} using boundaries {bucket_boundaries}")

        resolutions = np.array([dataset[i][0].shape[-2:] for i in range(len(dataset))]) # resolutions[i] = (h, w) of first item of dataset[i]
        self.logger.debug(f"Created resolutions list for dataset: {resolutions}")
        bucket_boundaries.append((float("inf"), float("inf"))) # append bucket for any leftovers not captured by largest passed size
        # buckets is a 2d array. Row j corresponds to a bucket whose boundary is bucket_boundaries[j]. Each row
        # contains the indices of the examples of dataset belonging to that bucket
        self.buckets = [[] for _ in range(len(bucket_boundaries))]
        self.logger.debug(f"Initialized buckets array: {self.buckets}")
        for i, res in enumerate(resolutions):
            for j, boundary in enumerate(bucket_boundaries):
                if res[0] <= boundary[0] and res[1] <= boundary[1]:
                    self.buckets[j].append(i) # append example index i to bucket j
                    break

        for i in range(len(self.buckets)):
            self.buckets[i] = np.array(self.buckets[i])
            if len(self.buckets[i]) == 0:
                self.logger.debug(f"Deleting buckets left empty")
                del self.buckets[i:] # if ith bucket has no images, any larger buckets also must have none
                break
        self.logger.debug(f"Finished creating buckets array: {self.buckets}")
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    # shuffling shuffles the order with which we access buckets (so it's not always smallest -> largest) and,
    # within each bucket, the order with which example indices are yielded
    def __iter__(self):
        bucket_order = np.arange(len(self.buckets))
        if self.shuffle:
            self.logger.debug("Shuffling indices")
            np.random.shuffle(bucket_order)
            self.logger.debug(f"New bucket order: {bucket_order}")

        for bucket_idx in bucket_order:
            bucket = self.buckets[bucket_idx]
            self.logger.debug(f"Chosen bucket: {bucket}")
            if self.shuffle:
                np.random.shuffle(bucket)
                self.logger.debug(f"Shuffling indices within bucket. New order: {bucket}")
            for i in range(0, len(bucket), self.batch_size):
                yield bucket[i:i + self.batch_size] # slicing accounts for trying to index out of bounds

PATCH_SIZE = 16 # actual patch will be PATCH_SIZE x PATCH_SIZE
AUGMENTATION_P = 0.5 # probability to apply camera augmentation 

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

    # for simplicity, just do bicubic interpolated 1d embeddings
    # bucketiterator since will have to pad to match longest sequence

    encoder = timm.create_model("convnextv2_base", pretrained=True)
    # pretrained data transfomr: {'input_size': (3, 224, 224), 'interpolation': 'bicubic', 'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 'crop_pct': 0.875, 'crop_mode': 'center'}
    """Image size isn't enforced, and ours is variable so specify a dummy size. As MAE paper states, decoder can 
    be lightweight. Number of attention heads is same as ViT_L but only use 2 blocks and halve
    hidden and MLP dim"""
    decoder = VisionTransformer(image_size=128, patch_size=PATCH_SIZE, num_layers=2, num_heads=16, hidden_dim=512, mlp_dim=2048)
    #print(decoder)