from datasets import GrandStaffLMXDataset, PreparedDataset, OlimpicDataset, GrandStaffPreTrainWrapper, OlimpicPreTrainWrapper, PreTrainWrapper
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils import GRAND_STAFF_ROOT_DIR, PRIMUS_PREPARED_ROOT_DIR, DOREMI_PREPARED_ROOT_DIR, OLIMPIC_SYNTHETIC_ROOT_DIR, OLIMPIC_SCANNED_ROOT_DIR
from pathlib import Path
import matplotlib.pyplot as plt
from torch.nn import Identity

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
    return widths, heights
    
# takes paths to height and width np arrays on disk and a patch size to get some stats using that patch size
# saves calculated arrays to stats_dir
def calc_patchify_stats(stats_dir, patch_size):
    heights = np.load(stats_dir / "image_heights.npy")
    widths = np.load(stats_dir / "image_widths.npy")

    aspect_ratios = widths / heights
    patchified_heights = heights / patch_size
    patchified_widths = widths / patch_size

    print(f"Aspect ratios\nmax: {aspect_ratios.max()}, min: {aspect_ratios.min()}\nPatchified heights\nmax: {patchified_heights.max()}, min: {patchified_heights.min()}\nPatchified widths\nmax: {patchified_widths.max()}, min: {patchified_widths.min()}")
    print(f"Saving np arrays to {stats_dir}")

    np.save((stats_dir / "aspect_ratios.npy"), aspect_ratios)
    np.save((stats_dir / "patchified_heights.npy"), patchified_heights)
    np.save((stats_dir / "patchified_widths.npy"), patchified_widths)

    fig, axs = plt.subplots(1, 3)
    fig.set_figheight(12)
    fig.set_figwidth(12)
    
    axs[0].set_title(f"Aspect ratios\nMin: {aspect_ratios.min()} Max: {aspect_ratios.max()}")
    axs[0].hist(aspect_ratios)
    axs[0].axvline(x=1, color="red", linestyle="--", linewidth=2)

    axs[1].set_title(f"Patchified heights\nMin: {patchified_heights.min()} Max: {patchified_heights.max()}")
    axs[1].hist(patchified_heights)

    axs[2].set_title(f"Patchified widths\nMin: {patchified_widths.min()} Max: {patchified_widths.max()}")
    axs[2].hist(patchified_widths)

    fig.savefig(stats_dir / "patchified_stats_plot.png")

if __name__ == "__main__":
    # distorted and undistorted images have different sizes, even for the same sheet
    grand_staff_pretrain_unaugmented = GrandStaffPreTrainWrapper(GRAND_STAFF)
    # augment_p = 1 and no-op transform to only measure distorted versions
    grand_staff_pretrain_augmented = GrandStaffPreTrainWrapper(GRAND_STAFF, augment_p=1, transform=Identity())
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

    all_distrs_widths = np.array([])
    all_distrs_heights = np.array([])
    for name, dataloader in pre_train_dataloaders.items():
        dataset_stats_dir = STATS_DIR / (name.replace(" ", "_").lower() + "_stats")
        dataset_stats_dir.mkdir(exist_ok=True)
        print(f"Creating image stats distribution for {name} dataset")
        widths, heights = calc_distrs(dataloader, dataset_stats_dir)
        all_distrs_widths = np.concat([all_distrs_widths, widths])
        all_distrs_heights = np.concat([all_distrs_heights, heights])

    all_distrs_dir = STATS_DIR / "pretrain_distr_stats"
    all_distrs_dir.mkdir(exist_ok=True)
    plot_and_save_stats(all_distrs_widths, all_distrs_heights, all_distrs_dir)
    np.save((all_distrs_dir / "image_widths.npy"), all_distrs_widths)
    np.save((all_distrs_dir / "image_heights.npy"), all_distrs_heights)