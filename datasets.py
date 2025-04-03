import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from PIL import Image

# superclass to deal with the commonalities between the LMX datasets
class LMXDataset(Dataset):
    def __init__(self, root_dir, split_file_name, transform=None):
        self.root_dir = Path(root_dir)
        split_file = self.root_dir / split_file_name
        self.id_df = pd.read_csv(split_file, header=None)
        self.transform = transform

    def __len__(self):
        return len(self.id_df)

"""This dataset class just deals with returning all the data in the GrandStaff LMX dataset that will be needed at
some point in the process, ie the original and distorted versions of each score and LMX file (as one string). 
For each stage, deciding which of these to use/modify will be done by a wrapper on top of this base dataset

root_dir should be to the GrandStaffLMX dataset root directory. The directory containing the actual
GrandStaff dataset with the images should be located within root_dir, ie at root_dir/grandstaff. Any transform
will be only applied to the images. split_file should be one of samples.test.txt, samples.train.txt, samples.dev.txt"""
class GrandStaffLMXDataset(LMXDataset):
    def __getitem__(self, idx):
        original_img_path = self.root_dir / "grandstaff" / (self.id_df.iat[idx, 0] + ".jpg") 
        distorted_img_path = self.root_dir / "grandstaff" / (self.id_df.iat[idx, 0] + "_distorted.jpg")
        original_img = Image.open(original_img_path)
        distorted_img = Image.open(distorted_img_path)

        if self.transform:       
            original_img = self.transform(original_img)
            distorted_img = self.transform(distorted_img)

        lmx_path = self.root_dir / (self.id_df.iat[idx, 0] + ".lmx")
        with open(lmx_path, "r") as f:
            lmx = f.read()

        return original_img, distorted_img, lmx

"""Base wrapper for MAE pre-training in the simplest case where a dataset instance returns
one item, the score image. Indexing this returns a tuple of input_img, target_img, allowing
for the input image to differ from the target. Transformations will be applied only to the
input image (so the autoencoder always tries to reconstruct the original image)"""
class PreTrainWrapper(Dataset):
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        input_img = self.base_dataset[idx]
        target_img = self.base_dataset[idx]
        if self.transform:
            input_img = self.transform(input_img)

        return input_img, target_img

"""Wrapper for MAE pre-training for Olimpic. Some slightly different logic is needed to handl
the lmx files Olimpic datasets have"""
class OlimpicPreTrainWrapper(PreTrainWrapper):
    def __getitem__(self, idx):
        input_img, _ = self.base_dataset[idx]
        target_img, _ = self.base_dataset[idx]
        if self.transform:
            input_img = self.transform(input_img)

        return input_img, target_img

"""Wrapper for MAE pre-training for GrandStaff. The dataset already contains images with some of the desired augmentations,
so augmenting the data needs different logic. With probability 1 - augment_p, return the original image as the input image.
Otherwise, apply the transformation to the distorted image retrieved from the dataset"""
class GrandStaffPreTrainWrapper(PreTrainWrapper):
    def __init__(self, base_dataset, augment_p=0, transform=None):
        if augment_p > 0:
            assert transform is not None, "Augmentation transform must be specified for non-zero augment_p"
        super().__init__(base_dataset, transform)
        self.augment_p = augment_p

    def __getitem__(self, idx):
        original_img, distorted_img, _ = self.base_dataset[idx]
        rand = torch.rand(1).item()
        if rand < self.augment_p: # augment distorted image and return it as the input image
            input_img = self.transform(distorted_img)
            return input_img, original_img
        else: # return original image as input image
            return original_img, original_img

# because Primus and DoReMi are both prepared to fit the same format, the same logic can be used for them
class PreparedDataset(Dataset):
    # root_dir should be the root directory created by the prepare script
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.id_df = pd.read_csv(self.root_dir / "ids.csv")
        self.transform = transform
    
    def __len__(self):
        return len(self.id_df)

    def __getitem__(self, idx):
        img_id = self.id_df.at[idx, "id"]
        img_path = self.root_dir / "images" / (img_id + ".png")
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img

# can either specify the synthetic or scanned dataset
class OlimpicDataset(LMXDataset):
    def __getitem__(self, idx):
        img_path = self.root_dir / (self.id_df.iat[idx, 0] + ".png")
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        lmx_path = self.root_dir / (self.id_df.iat[idx, 0] + ".lmx")
        with open(lmx_path, "r") as f:
            lmx = f.read()

        return img, lmx

# TODO: olimpic dataset (scanned + synthetic)

# TODO: create pretrain wrapper for grandstaff that always returns original image as target and then
# either returns original image is input image to mask, returns the distorted image as input image,
# or phone augments the original image to be input image (should this be a custom transformation?)


# all pre train datasets shoudl return a potentially augmented image to mask and then the target image 
# which should always be un augmented
# honestly for grandstaff may not need to separate distorted from camera augmentation. distortion could just
# be start of camera augmentation (and then maybe just add perspective shift, brightness, color jitter)