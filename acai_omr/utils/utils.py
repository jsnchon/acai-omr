import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch.utils.data.sampler import BatchSampler
from torchvision.transforms import v2, InterpolationMode
import torchvision.transforms.v2.functional as F
import logging
import math
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from acai_omr.models.models import MAELoss, OMRCELoss
from acai_omr.config import LMX_EOS_TOKEN
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

# GRPO stuff
@dataclass
class RolloutConfig:
    group_size: int
    max_actions: int
    top_k: int
    temperature: float

@dataclass
class RewardComponents:
    tedn_scores: torch.Tensor | float
    wellformedness_scores: torch.Tensor | float
    f1_scores: torch.Tensor | float
    repeat_penalty: torch.Tensor | float
    len_penalty: torch.Tensor | float

    def __add__(self, other):
        return RewardComponents(
            self.tedn_scores + other.tedn_scores,
            self.wellformedness_scores + other.wellformedness_scores,
            self.f1_scores + other.f1_scores,
            self.repeat_penalty + other.repeat_penalty,
            self.len_penalty + other.len_penalty,
        )

    def __truediv__(self, divisor):
        return RewardComponents(
            self.tedn_scores / divisor,
            self.wellformedness_scores / divisor,
            self.f1_scores / divisor,
            self.repeat_penalty / divisor,
            self.len_penalty / divisor,
        )

    # if instance variables hold (R, ) tensors, each of those tensors is averaged to get components
    # averaged across all rollouts into singular floats
    def avg_over_rollouts(self):
        return RewardComponents(
            tedn_scores = self.tedn_scores.mean().item(),
            wellformedness_scores = self.wellformedness_scores.mean().item(),
            f1_scores = self.f1_scores.mean().item(),
            repeat_penalty = self.repeat_penalty.mean().item(),
            len_penalty = self.len_penalty.mean().item(),
        )

    def to_dict(self):
        return {
            "tedn_scores": self.tedn_scores,
            "wellformedness_scores" : self.wellformedness_scores,
            "f1_scores" : self.f1_scores,
            "repeat_penalty" : self.repeat_penalty,
            "len_penalty" : self.len_penalty,
        }
    
@dataclass
class RewardConfig:
    lambda_tedn: float
    lambda_well_formed: float
    lambda_f1: float
    lambda_repeat: float
    lambda_len: float
    alpha_tedn: float
    alpha_well_formed: float
    gamma: float
    delta: int
    tau: int

@dataclass
class LossConfig:
    entropy_beta: float
    lambda_ce: float

@dataclass
class UpdateConfig:
    epsilon: float
    update_epochs: int
    max_grad_norm: float

@dataclass
class GRPOConfig:
    rollout_config: RolloutConfig
    reward_config: RewardConfig
    loss_config: LossConfig
    update_config: UpdateConfig
    mini_validation_freq: int
    checkpoint_freq: int

    def get_configs(self):
        return self.rollout_config, self.reward_config, self.loss_config, self.update_config

@dataclass
class StepCounter:
    # we log at many different granularities, but keep track of one global step to align everything. Each
    # optimizer step is a global step
    global_step: int

class GRPOLogger:
    def __init__(self, writer, epoch_stats_df):
        self.writer = writer
        self.epoch_stats_df = epoch_stats_df

    # only log what's changing over time
    def log_configs(self, grpo_config, global_step):
        rollout_config, reward_config, loss_config, _ = grpo_config.get_configs()
        prefix = "train/config"

        self.writer.add_scalar(f"{prefix}/max_actions", rollout_config.max_actions, global_step)
        self.writer.add_scalar(f"{prefix}/top_k", rollout_config.top_k, global_step)
        self.writer.add_scalar(f"{prefix}/temperature", rollout_config.temperature, global_step)
        self.writer.add_scalar(f"{prefix}/lambda_len", reward_config.lambda_len, global_step)
        self.writer.add_scalar(f"{prefix}/entropy_beta", loss_config.entropy_beta, global_step)
        self.writer.add_scalar(f"{prefix}/lambda_ce", loss_config.lambda_ce, global_step)

    def log_raw_reward_components(self, reward_components, global_step):
        prefix = f"train/reward/raw/components"
        
        reward_components = reward_components.avg_over_rollouts()
        components_dict = reward_components.to_dict()
        self.writer.add_scalars(prefix, components_dict, global_step)
  
    def log_raw_reward_stats(self, raw_group_rewards, global_step):
        prefix = f"train/reward/raw/overall"
        # taking stats over whole tensor gives raw reward stats over all rollouts
        self.writer.add_scalar(f"{prefix}/mean", raw_group_rewards.mean().item(), global_step)
        self.writer.add_scalar(f"{prefix}/std", raw_group_rewards.std().item(), global_step)
        self.writer.add_scalar(f"{prefix}/min", raw_group_rewards.min().item(), global_step)
        self.writer.add_scalar(f"{prefix}/max", raw_group_rewards.max().item(), global_step)

    def log_group_advantages(self, advantages, global_step):
        prefix = f"train/reward/advantage"

        self.writer.add_scalar(f"{prefix}/min", advantages.min().item(), global_step)
        self.writer.add_scalar(f"{prefix}/max", advantages.max().item(), global_step)

    def log_raw_objective_components(self, grpo_objective, entropy_bonus, ce_loss, global_step):
        prefix = f"train/objective/raw/components"

        self.writer.add_scalar(f"{prefix}/grpo", grpo_objective.item(), global_step)
        self.writer.add_scalar(f"{prefix}/entropy_bonus", entropy_bonus.item(), global_step)
        self.writer.add_scalar(f"{prefix}/ce_loss", ce_loss.item(), global_step)

    def log_overall_loss(self, overall_loss, global_step):
        self.writer.add_scalar("train/loss", overall_loss.item(), global_step)

    def log_mini_validation_stats(self, mini_val_reward, mini_val_reward_components, mini_val_ce_loss, global_step):
        prefix = f"mini_val"

        self.writer.add_scalar(f"{prefix}/reward", mini_val_reward, global_step)
        components_dict = mini_val_reward_components.to_dict()
        self.writer.add_scalars(f"{prefix}/reward/components", components_dict, global_step)
        self.writer.add_scalar(f"{prefix}/ce_loss", mini_val_ce_loss, global_step)

    def log_epoch_level_stats(self, epoch_stats: dict, global_step):
        prefix = f"epoch"

        self.writer.add_scalar(f"{prefix}/train/loss/overall", epoch_stats["avg_train_overall_loss"], global_step)
        self.writer.add_scalar(f"{prefix}/train/reward", epoch_stats["avg_train_reward"], global_step)
        self.writer.add_scalars(f"{prefix}/train/reward/components", epoch_stats["avg_train_reward_components"].to_dict(), global_step)
        self.writer.add_scalar(f"{prefix}/train/loss/ce", epoch_stats["avg_train_ce_loss"], global_step)

        self.writer.add_scalar(f"{prefix}/val/reward", epoch_stats["full_val_reward"], global_step)
        self.writer.add_scalars(f"{prefix}/val/reward/components", epoch_stats["full_val_reward_components"].to_dict(), global_step)
        self.writer.add_scalar(f"{prefix}/val/loss/ce", epoch_stats["full_val_ce_loss"], global_step)

    def update_epoch_stats_df(self, epoch_stats: dict, epoch: int):
        self.epoch_stats_df.loc[epoch] = epoch_stats

    def log_lr(self, optimizer, global_step):
        prefix = "train"
        self.writer.add_scalar(f"{prefix}/lr", optimizer.param_groups[0]["lr"], global_step)

    def flush(self, csv_path):
        self.writer.flush()
        self.epoch_stats_df.to_csv(csv_path)

# convert a (1, T) tensor of lmx token indices into a single lmx string. This assumes the sequence starts with <bos> (doesn't
# have to end with <eos>, eg if it was truncated)
def stringify_lmx_seq(lmx_seq: torch.Tensor, idxs_to_tokens: dict[int, str]):
    lmx_seq = [idxs_to_tokens[idx.item()] for idx in lmx_seq]
    if lmx_seq[-1] == LMX_EOS_TOKEN:
        lmx_seq.pop(-1)
    lmx_seq = lmx_seq[1: ]
    lmx_seq = " ".join(lmx_seq)
    return lmx_seq

def stepwise_cosine_anneal_with_warmup(optimizer, warmup_steps, total_epochs, final_lr, num_steps_per_epoch):
    warmup = LinearLR(optimizer, start_factor=5e-3, end_factor=1.0, total_iters=warmup_steps)
    anneal_total_iters = total_epochs * num_steps_per_epoch - warmup_steps
    anneal = CosineAnnealingLR(optimizer, T_max=anneal_total_iters, eta_min=final_lr)
    return SequentialLR(optimizer, schedulers=[warmup, anneal], milestones=[warmup_steps])

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

# collate ragged batch into a list of (input, target) tensors for the MAE/teacher forcing logic to handle
def ragged_collate_fn(batch):
    collated_batch = []
    for example in batch:
        collated_batch.append(example)
    return collated_batch

def basic_fig_setup(title, num_epochs, y_label):
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(12)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)
    ax.grid()
    x_axis = np.arange(1, num_epochs + 1) # make sure each stat is lined up with integer epoch labels
    return fig, ax, x_axis

def graph_losses(train_losses, validation_losses, plot_file_path):
    fig, ax, x_axis = basic_fig_setup("Training stats", len(train_losses), "Average loss")
    ax.plot(x_axis, train_losses, label="Train loss", color="blue")
    ax.plot(x_axis, validation_losses, label="Validation loss", color="red")
    ax.legend()
    print(f"Saving plot to {plot_file_path}")
    fig.savefig(plot_file_path)
    plt.close(fig)

def graph_lrs_no_ft(epoch_lrs, lr_plot_file_path):
    fig, ax, x_axis = basic_fig_setup("Learning rates over time", len(epoch_lrs), "Learning rate (at epoch start)")
    ax.plot(x_axis, epoch_lrs)
    print(f"Saving plot to {lr_plot_file_path}")
    fig.savefig(lr_plot_file_path)
    plt.close(fig)

def graph_lrs_ft(base_lrs, fine_tune_lrs, lr_plot_file_path):
    fig, ax, x_axis = basic_fig_setup("Learning rates over time", len(base_lrs), "Learning rate (at epoch start)")
    ax.plot(x_axis, base_lrs, label="Base lr", color="blue")
    ax.plot(x_axis, fine_tune_lrs, label="Fine-tune lr", color="red")
    print(f"Saving plot to {lr_plot_file_path}")
    fig.savefig(lr_plot_file_path)
    plt.close(fig)

# saves everything to stats directory created in pretrain setup
def save_pre_train_stats(stats_dir_path, epoch_training_losses, epoch_validation_losses, epoch_lrs):
    loss_plot_path = stats_dir_path / "losses.png"
    lr_plot_path = stats_dir_path / "lrs.png"
    csv_path = stats_dir_path / "training_stats.csv"
    graph_losses(epoch_training_losses, epoch_validation_losses, loss_plot_path)
    graph_lrs_no_ft(epoch_lrs, lr_plot_path)
    stats_df = pd.DataFrame({
        "Epoch": np.arange(1, len(epoch_lrs) + 1),
        "Training loss": epoch_training_losses,
        "Validation loss": epoch_validation_losses,
        "Lr at start": epoch_lrs,
    })
    print(f"Writing training stats csv to {csv_path}")
    stats_df.to_csv(csv_path)

def graph_tf_probs(tf_probs, plot_path):
    fig, ax, x_axis = basic_fig_setup("Teacher forcing probabilities", len(tf_probs), "Probability")
    ax.plot(x_axis, tf_probs)
    print(f"Saving plot to {plot_path}")
    fig.savefig(plot_path)
    plt.close(fig)

def graph_taus(taus, plot_path):
    fig, ax, x_axis = basic_fig_setup("Gumbel-softmax taus", len(taus), "Tau")
    ax.plot(x_axis, taus)
    print(f"Saving plot to {plot_path}")
    fig.savefig(plot_path)
    plt.close(fig)

# saves everything to stats directory created in pretrain setup
def save_teacher_force_training_stats(stats_dir_path, train_stats_df: pd.DataFrame):
    graph_losses(train_stats_df["Train loss"], train_stats_df["Validation loss"], (stats_dir_path / "losses.png"))
    graph_lrs_ft(train_stats_df["Base lr at start"], train_stats_df["Fine-tune lr at start"], (stats_dir_path / "lrs.png"))
    graph_tf_probs(train_stats_df["Teacher forcing probability"], (stats_dir_path / "tf_probs.png"))
    graph_taus(train_stats_df["Gumbel-softmax tau"], (stats_dir_path / "taus.png"))

    csv_path = stats_dir_path / "training_stats.csv"
    print(f"Writing training stats csv to {csv_path}")
    train_stats_df.to_csv(csv_path)

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

    loss_fn = OMRCELoss(pad_idx=vitomr.decoder.pad_idx, label_smoothing=0.0)
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

# previous code wrote loss and validation curves to separate plots. This little utility function will use the stats
# csv to reformat the training stats to the new plot versions. The args should be pathlib Path instances
def reformat_training_stats(old_stats_dir_path, new_stats_dir_path):
    stats_df_path = old_stats_dir_path / "training_stats.csv"
    stats_df = pd.read_csv(stats_df_path)
    training_losses = stats_df["Training loss"]
    validation_losses = stats_df["Validation loss"]
    lrs = stats_df["Lr at start"]

    save_pre_train_stats(new_stats_dir_path, training_losses, validation_losses, lrs)

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
