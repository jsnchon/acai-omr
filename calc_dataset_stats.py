# todo: go through each dataset, calc distr for image sizes
from datasets import GrandStaffLMXDataset, PreparedDataset, OlimpicDataset, GrandStaffPreTrainWrapper, OlimpicPreTrainWrapper, PreTrainWrapper
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils import GRAND_STAFF_ROOT_DIR, PRIMUS_PREPARED_ROOT_DIR, DOREMI_PREPARED_ROOT_DIR, OLIMPIC_SYNTHETIC_ROOT_DIR, OLIMPIC_SCANNED_ROOT_DIR
from pathlib import Path
import matplotlib.pyplot as plt

GRAND_STAFF = GrandStaffLMXDataset(GRAND_STAFF_ROOT_DIR, "samples.train.txt", transform=ToTensor())
PRIMUS_PREPARED = PreparedDataset(PRIMUS_PREPARED_ROOT_DIR, transform=ToTensor())
DOREMI_PREPARED = PreparedDataset(DOREMI_PREPARED_ROOT_DIR, transform=ToTensor())
OLIMPIC_SYNTHETIC = OlimpicDataset(OLIMPIC_SYNTHETIC_ROOT_DIR, "samples.train.txt", transform=ToTensor())

def plot_and_save_stats(widths, heights, dataset_stats_dir):
    fig, axs = plt.subplots(1, 2)
    fig.set_figheight(12)
    fig.set_figwidth(12)
    fig.suptitle(f"Image widths and heights")
    fig.supxlabel("Pixels")
    fig.supylabel("Frequency")

    widths_mean = np.mean(widths)
    widths_median = np.median(widths)
    axs[0].set_title(f"Widths\nMean: {widths_mean} Median: {widths_median}")
    axs[0].hist(widths)

    heights_mean = np.mean(heights)
    heights_median = np.median(heights)
    axs[1].set_title(f"Heights\nMean: {heights_mean} Median: {heights_median}")
    axs[1].hist(heights)
    fig.savefig(dataset_stats_dir / "stats_plot.png") 

    np.save((dataset_stats_dir / "image_widths.npy"), widths)
    np.save((dataset_stats_dir / "image_heights.npy"), heights)
 
def calc_distrs(dataloader, dataset_stats_dir):
    widths = []
    heights = []
    # set of all input_img contains all target_imgs, so ignore returned target_img
    for input_img, _ in tqdm(dataloader):
        # using dataloader, so input_img is a 1 x C x H x W tensor 
        # use DataLoader because iterating through Dataset objects doesn't respect __len__
        input_img = input_img.squeeze()
        widths.append(input_img.shape[2])
        heights.append(input_img.shape[1])
    
    plot_and_save_stats(widths, heights, dataset_stats_dir)

# distorted and undistorted images have different sizes, even for the same sheet
grand_staff_pretrain_unaugmented = GrandStaffPreTrainWrapper(GRAND_STAFF)
grand_staff_pretrain_augmented = GrandStaffPreTrainWrapper(GRAND_STAFF)
primus_pretrain = PreTrainWrapper(PRIMUS_PREPARED)
doremi_pretrain = PreTrainWrapper(DOREMI_PREPARED)
# only synthetic dataset has training examples
olimpic_pretrain = OlimpicPreTrainWrapper(OLIMPIC_SYNTHETIC)
pre_train_dataloaders = {
    "GrandStaff unaugmented": DataLoader(grand_staff_pretrain_unaugmented), 
    "GrandStaff augmented": DataLoader(grand_staff_pretrain_augmented), 
    "Primus": DataLoader(primus_pretrain), 
    "DoReMi": DataLoader(doremi_pretrain), 
    "Olimpic synthetic": DataLoader(olimpic_pretrain),
    }
STATS_DIR = Path("data/dataset_stats")
STATS_DIR.mkdir(exist_ok=True)

for name, dataloader in pre_train_dataloaders.items():
    dataset_stats_dir = STATS_DIR / (name.replace(" ", "_").lower() + "_stats")
    dataset_stats_dir.mkdir(exist_ok=True)
    print(f"Creating image stats distribution for {name} dataset")
    calc_distrs(dataloader, dataset_stats_dir)