import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from PIL import Image

"""This dataset class just deals with returning all the data in the GrandStaff LMX dataset that will be needed at
some point in the process, ie the original and distorted versions of each score and LMX file (as one string). 
For each stage, deciding which of these to use/modify will be done by a wrapper on top of this base dataset

root_dir should be to the GrandStaffLMX dataset root directory. The directory containing the actual
GrandStaff dataset with the images should be located within root_dir, ie at root_dir/grandstaff. Any transform
will be only applied to the images. split_file should be one of samples.test.txt, samples.train.txt, samples.dev.txt"""
class GrandStaffLMXDataset(Dataset):
    def __init__(self, root_dir, split_file_name, transform=None):
        self.root_dir = Path(root_dir)
        split_file = self.root_dir / split_file_name
        self.id_df = pd.read_csv(split_file, header=None)
        self.transform = transform

    def __len__(self):
        return len(self.id_df)

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

# TODO: olimpic dataset (scanned + synthetic)

# TODO: create pretrain wrapper for grandstaff that always returns original image as target and then
# either returns original image is input image to mask, returns the distorted image as input image,
# or phone augments the original image to be input image (should this be a custom transformation?)


# all pre train datasets shoudl return a potentially augmented image to mask and then the target image 
# which should always be un augmented
# honestly for grandstaff may not need to separate distorted from camera augmentation. distortion could just
# be start of camera augmentation (and then maybe just add perspective shift, brightness, color jitter)