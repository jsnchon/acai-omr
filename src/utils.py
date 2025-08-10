import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch.utils.data.sampler import BatchSampler
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as F
import logging
import math
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from models import MAELoss, OMRLoss
import pandas as pd
from pathlib import Path

# num_train_batches is the number of batches in each epoch. If passed, the scheduler will configure to be called
# each minibatch instead of each epoch
def cosine_anneal_with_warmup(optimizer, warmup_epochs, total_epochs, final_lr, num_train_batches=None):
    if not num_train_batches:
        warmup = LinearLR(optimizer, start_factor=5e-3, end_factor=1.0, total_iters=warmup_epochs)
        anneal = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=final_lr)
        return SequentialLR(optimizer, schedulers=[warmup, anneal], milestones=[warmup_epochs])
    else:
        warmup_total_iters = warmup_epochs * num_train_batches
        anneal_total_iters = (total_epochs - warmup_epochs) * num_train_batches
        warmup = LinearLR(optimizer, start_factor=5e-3, end_factor=1.0, total_iters=warmup_total_iters)
        anneal = CosineAnnealingLR(optimizer, T_max=anneal_total_iters, eta_min=final_lr)
        return SequentialLR(optimizer, schedulers=[warmup, anneal], milestones=[warmup_total_iters])

# collate ragged batch into a list of (input, target) tensors for the MAE logic to handle
def ragged_collate_fn(batch):
    collated_batch = []
    for example in batch:
        collated_batch.append((example[0], example[1]))
    return collated_batch

def plot_lr_schedule(scheduler, optimizer, num_epochs):
    lrs = []
    for _ in range(num_epochs):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
        # train code would go here
    
    plt.plot(lrs)
    plt.title("Learning rate over time using scheduler")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.grid(True)
    plt.savefig("lr_over_epochs.png")

# shows the input, mae's reconstruction, and target images using one example ex side by side
def show_mae_prediction(mae, ex, patch_size, save_path: str):
    loss_fn = MAELoss()
    mae.eval()
    with torch.no_grad():
        pred, loss_mask, loss_target = mae([ex])
    loss = loss_fn(pred, loss_mask, loss_target)
    input = ex[0]
    target = ex[1]
    fold = nn.Fold(output_size=(input.shape[1], input.shape[2]), kernel_size=patch_size, stride=patch_size)
    pred = fold(pred.transpose(1, 2)) # fold back into 1 x C x H x W image

    num_channels = pred.shape[1]
    if num_channels == 1:
        fig, axs = plt.subplots(1, 3)
        fig.set_figheight(4)
        fig.set_figwidth(16)
        fig.suptitle(f"Loss: {loss}")
        axs[0].imshow(input.cpu().squeeze(0), cmap="gray") # have to move tensors to CPU so matplot can convert them to numpy
        axs[0].set_title("Input image")
        axs[1].imshow(pred.cpu().detach().squeeze(0).squeeze(0), cmap="gray")
        axs[1].set_title("MAE reconstruction prediction")
        axs[2].imshow(target.cpu().squeeze(0), cmap="gray")
        axs[2].set_title("Target image")
        print(f"Saving prediction to {save_path}")
        fig.savefig(save_path)
    else:
        raise NotImplementedError("Images are assumed to be grayscale")

def show_vitomr_prediction(vitomr, ex, sample_save_dir: str):
    sample_save_dir = Path(sample_save_dir)
    sample_save_dir.mkdir(exist_ok=True)

    loss_fn = OMRLoss(padding_idx=vitomr.decoder.padding_idx)
    vitomr.eval()
    with torch.no_grad():
        pred, target_seq = vitomr([ex])
    loss = loss_fn(pred, target_seq)

    idxs_to_tokens = vitomr.decoder.idxs_to_tokens
    pred = torch.softmax(pred, dim=-1)
    pred = torch.argmax(pred, dim=-1)
    pred = pred.squeeze(0)
    pred = [idxs_to_tokens[idx.item()] for idx in pred]
    pred = " ".join(pred)

    target_seq = target_seq.squeeze(0)
    target_seq = [idxs_to_tokens[idx.item()] for idx in target_seq]
    target_seq = " ".join(target_seq)

    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(12)
    fig.suptitle(f"Sequences cross entropy loss: {loss}")
    ax.imshow(ex[0].cpu().squeeze(0), cmap="gray")
    ax.set_title("Input image")

    plot_save_path = sample_save_dir / "input_image.png"
    print(f"Saving input image to {plot_save_path}")
    fig.savefig(plot_save_path)

    pred_file_path = sample_save_dir / "pred.txt"
    print(f"Saving predicted token sequence to {pred_file_path}")
    with open(pred_file_path, "w") as f:
        f.write(pred)

    target_seq_file_path = sample_save_dir / "target_seq.txt"
    print(f"Saving target token sequence to {target_seq_file_path}")
    with open(target_seq_file_path, "w") as f:
        f.write(target_seq)

# dataset should be initialized with a ToTensor transformation for any images
def display_dataset_img(dataset, index, save_path): 
    data = dataset[index]

    if isinstance(data, tuple):
        img_tensors = []
        for item in data:
            if torch.is_tensor(item):
                img_tensors.append(item)

        rows = len(img_tensors)
        COLS = 1
        if rows == 0:
            print("No images retrieved to show. Make sure the dataset was initialized with a ToTensor transformation")
            return

        fig, axs = plt.subplots(rows, COLS)
        if rows == 1: # tuple contains one tensor, so one subplot and axs is just one axes object, not a list
            axs.imshow(img_tensors[0].squeeze(0), cmap="gray")
        else:
            # one image per col
            for i in range(rows):
                ax = axs[i]
                ax.imshow(img_tensors[0].squeeze(0), cmap="gray")
    else:
        fig, ax = plt.subplots()
        ax.imshow(img_tensors[0].squeeze(0), cmap="gray")

    fig.suptitle(f"Index: {index}")
    plt.savefig(save_path)
    return data

def sample_pre_train_dataset(pre_train_dataset, num_samples, patch_size):
    sample_indices = np.floor(np.random.rand(num_samples) * len(pre_train_dataset)).astype(int)
    for i in range(num_samples):
        idx = sample_indices[i]
        input_img, _ = display_dataset_img(pre_train_dataset, idx)
        print(f"Input image shape: {input_img.shape}")
        width_patches = input_img.shape[1] // patch_size
        height_patches = input_img.shape[2] // patch_size
        print(f"Patch dimensions\nWidth: {width_patches} patches, Height: {height_patches}, Total: {width_patches * height_patches} patches")

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

# resize image to a nearby patch divisible resolution that also, when patchified, has a sequence length
# within a certain budget (to avoid excessively long input sequences)
class DynamicResize(nn.Module):
    def __init__(self, patch_size, max_seq_len, pe_max_height, pe_max_width, crop_imgs):
        super().__init__()
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.pe_max_height = pe_max_height
        self.pe_max_width = pe_max_width
        self.crop_imgs = crop_imgs # whether to center crop images larger than the max dims

    # img should be a tensor of shape C x H x W
    def forward(self, img):
        height = img.shape[-2]
        width = img.shape[-1]
        if width > height:
            aspect_ratio = width // height
            target_height = self.patch_size * math.floor(math.sqrt(self.max_seq_len / aspect_ratio))
            target_width = target_height * aspect_ratio
        else:
            aspect_ratio = height // width
            target_width = self.patch_size * math.floor(math.sqrt(self.max_seq_len / aspect_ratio))
            target_height = target_width * aspect_ratio

        img = F.resize(
            img,
            size=(target_height, target_width),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True
        )

        if self.crop_imgs:
            # center crop each of height and/or width if they're too large for the positional embedding
            if target_height / self.patch_size > self.pe_max_height:
                img = v2.functional.center_crop(img, (self.pe_max_height * self.patch_size, img.shape[-1]))
            if target_width / self.patch_size > self.pe_max_width:
                img = v2.functional.center_crop(img, (img.shape[-2], self.pe_max_width * self.patch_size))

        return img.clamp(0.0, 1.0)
    
def graph_model_stats(train_losses, validation_losses, plot_file_path):
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(12)
    ax.set_title(f"Training stats")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Average loss")
    x_axis = np.arange(1, len(train_losses) + 1) # make sure each stat is lined up with integer epoch labels
    ax.plot(x_axis, train_losses, label="Train loss", color="blue")
    ax.plot(x_axis, validation_losses, label="Validation loss", color="red")
    ax.grid()
    ax.legend()
    print(f"Saving plot to {plot_file_path}")
    fig.savefig(plot_file_path)
    plt.close(fig)

def graph_lrs(epoch_lrs, plot_file_path, fine_tuning):
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(12)
    ax.set_title("Learning rates over time")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate (at epoch start)")
    ax.grid()
    x_axis = np.arange(1, len(epoch_lrs) + 1)
    if fine_tuning:
        base_lrs, fine_tune_lrs = zip(*epoch_lrs)
        ax.plot(x_axis, list(base_lrs), label="Base lr", color="blue")
        ax.plot(x_axis, list(fine_tune_lrs), label="Fine-tune lr", color="red")
        ax.legend()
    else:
        ax.plot(x_axis, epoch_lrs)
    print(f"Saving plot to {plot_file_path}")
    fig.savefig(plot_file_path)
    plt.close(fig)

def graph_grad_norms(epoch_grad_norm_snapshots, plot_file_path):
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(12)
    ax.set_title("Gradient norms")
    ax.set_xlabel("Step count")
    ax.set_ylabel("L2 norm")
    ax.grid()
    x_axis = np.arange(1, epoch_grad_norm_snapshots[-1].step_num)
    encoder_norms = [snapshot.encoder_norm for snapshot in epoch_grad_norm_snapshots]
    transition_head_norms = [snapshot.transition_head_norm for snapshot in epoch_grad_norm_snapshots]
    decoder_norms = [snapshot.decoder_norm for snapshot in epoch_grad_norm_snapshots]
    ax.plot(x_axis, encoder_norms, label="Encoder", color="blue")
    ax.plot(x_axis, transition_head_norms, label="Transition head", color="orange")
    ax.plot(x_axis, decoder_norms, label="Decoder", color="purple")
    ax.legend()
    print(f"Saving plot to {plot_file_path}")
    fig.savefig(plot_file_path)
    plt.close(fig)
 
# saves everything to stats directory created in pretrain setup
def save_training_stats(stats_dir_path, epoch_training_losses, epoch_validation_losses, epoch_lrs, epoch_grad_norm_snapshots, fine_tuning=False):
    loss_plot_path = stats_dir_path / "losses.png"
    lr_plot_path = stats_dir_path / "lrs.png"
    grad_norms_plot_path = stats_dir_path / "grad_norms.png"
    csv_path = stats_dir_path / "training_stats.csv"
    graph_model_stats(epoch_training_losses, epoch_validation_losses, loss_plot_path)
    graph_lrs(epoch_lrs, lr_plot_path, fine_tuning)
    graph_grad_norms(epoch_grad_norm_snapshots, grad_norms_plot_path)
    if fine_tuning:
        base_lrs, fine_tune_base_lrs = zip(*epoch_lrs)
        stats_df = pd.DataFrame({
            "Epoch": np.arange(1, len(epoch_lrs) + 1),
            "Training loss": epoch_training_losses,
            "Validation loss": epoch_validation_losses,
            "Base lr at start": list(base_lrs),
            "Fine-tune base lr at start": list(fine_tune_base_lrs),
        })
        print(f"Writing training stats csv to {csv_path}")
        stats_df.to_csv(csv_path)
    else:
        stats_df = pd.DataFrame({
            "Epoch": np.arange(1, len(epoch_lrs) + 1),
            "Training loss": epoch_training_losses,
            "Validation loss": epoch_validation_losses,
            "Lr at start": epoch_lrs,
        })
        print(f"Writing training stats csv to {csv_path}")
        stats_df.to_csv(csv_path)

# previous code wrote loss and validation curves to separate plots. This little utility function will use the stats
# csv to reformat the training stats to the new plot versions. The args should be pathlib Path instances
def reformat_training_stats(old_stats_dir_path, new_stats_dir_path):
    stats_df_path = old_stats_dir_path / "training_stats.csv"
    stats_df = pd.read_csv(stats_df_path)
    training_losses = stats_df["Training loss"]
    validation_losses = stats_df["Validation loss"]
    lrs = stats_df["Lr at start"]

    save_training_stats(new_stats_dir_path, training_losses, validation_losses, lrs)

"""
DataLoader batching requires non-ragged batches, ie all examples are of the same length. A collate function to pad based on
the largest image in the batch will deal with this, but it's much more efficient if we ensure all batches have similarly sized
images to prevent over-padding. BucketBatchSampler takes a Dataset instance dataset and list of tuples bucket_boundaries and
groups examples within dataset together based on image size. Each tuple in bucket_boundaries specifies a max_h, max_w for a bucket. 
Images will be placed in the smallest bucket they can fit in. bucket_boundaries must be in ascending order of specified size

Note that this sampler buckets only based on the size of the first item returned by an indexing of dataset. The
assumption is this is always an image, and that any other images retrieved with it are of a similar size.
"""
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

        self.buckets = [np.array(bucket) for bucket in self.buckets if len(bucket) > 0]
        self.logger.debug(f"Finished creating buckets array: {self.buckets}")
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = dataset

    def __len__(self):
        return -(len(self.dataset) // -self.batch_size) # ceiling division

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
