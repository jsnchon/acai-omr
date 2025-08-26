import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import ConcatDataset, DataLoader
from torch.multiprocessing import Pool # regular multiprocessing complains about torch's multi-threadedness
from acai_omr.train.omr_teacher_force_train import set_up_omr_teacher_force_train
from acai_omr.train.datasets import GrandStaffLMXDataset, GrandStaffOMRTrainWrapper, OlimpicDataset 
from acai_omr.models.models import GRPOViTOMR
from acai_omr.utils.utils import stringify_lmx_seq, ragged_collate_fn
from acai_omr.config import GRAND_STAFF_ROOT_DIR, OLIMPIC_SYNTHETIC_ROOT_DIR, OLIMPIC_SCANNED_ROOT_DIR
from torchvision.transforms import v2, InterpolationMode
from olimpic_app.evaluation.TEDn_lmx_xml import TEDn_lmx_xml
from dataclasses import dataclass
import copy

AUGMENTATION_P = 0.3

BATCH_SIZE = 2
NUM_WORKERS = 6

NUM_ROLLOUTS_PER_IMG = 4
GRPO_EPOCHS = 6

TEACHER_FORCED_STATE_DICT_PATH = ""
# make sure to set include_musicxml = True for the datasets

@dataclass
class RolloutConfig:
    num_rollouts_per_img: int
    max_actions: int
    top_k: int
    temperature: float

@dataclass
class RewardComponents:
    tedn_scores: torch.Tensor
    wellformedness_scores: torch.Tensor
    f1_scores: torch.Tensor
    repeat_penalty: torch.Tensor
    len_penalty: torch.Tensor

    def __str__(self):
        return f"Max actions: {self.max_actions}\nTop_k: {self.top_k}\nSoftmax temperature: {self.temperature}"
    
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

# broadcast sequences across rollouts into an (R x T) padded tensor
def expand_target_lmx_seqs(target_lmx_seqs: tuple[torch.Tensor], num_rollouts_per_img, pad_idx, device):
    target_lmx_seqs = torch.nested.as_nested_tensor(target_lmx_seqs, device=device, layout=torch.jagged)
    target_lmx_seqs = target_lmx_seqs.to_padded_tensor(padding=pad_idx)
    target_lmx_seqs = target_lmx_seqs.unsqueeze(1)
    target_lmx_seqs = target_lmx_seqs.expand(-1, num_rollouts_per_img, -1)
    target_lmx_seqs = target_lmx_seqs.flatten(start_dim=0, end_dim=1)
    return target_lmx_seqs

# all these reward components should return (R, ) tensors, one component per rollout each

# target_musicxml_strs is a tuple where target_musicxml_strs[i] = the target musicxml for image i in the minibatch
# Returns three tensors of metrics across all rollouts, one for raw edit_costs, one for a bool on if a castrophic error 
# happened, and one for counts of minor delinearization errors
def calc_edit_costs(rollouts, pad_idx, batch_size, num_rollouts_per_img, target_musicxml_strs: tuple, idxs_to_tokens, num_parallel_processes=12):
    # unfold into individual groups since we can't broadcast the musicxml strings within groups
    rollout_groups = rollouts.view(batch_size, num_rollouts_per_img, rollouts.shape[-1])
    tedn_call_args = []
    for i, group in enumerate(rollout_groups):
        for rollout in group:
            rollout = rollout[~(rollout == pad_idx)] # ignore <pad> tokens
            predicted_lmx = stringify_lmx_seq(rollout, idxs_to_tokens)
            tedn_call_args.append((predicted_lmx, target_musicxml_strs[i], "lmx")) # share each target str across the whole group

    with Pool(processes=num_parallel_processes) as pool:
        results = pool.starmap( # list of (TEDnResult.edit_cost, catastrophic_errs, minor_errs) tuples, one per rollout
            TEDn_lmx_xml, tedn_call_args
        )
    
    # note that TEDn_lmx_xml() by default returns TEDnResult instances which store a gold_cost that could be
    # used for score normalization
    edit_costs, catastrophic_errors, minor_errors = zip(*results)
    edit_costs = torch.tensor(edit_costs, device=rollouts.device)
    catastrophic_errors = torch.tensor(catastrophic_errors, device=rollouts.device)
    minor_errors = torch.tensor(minor_errors, device=rollouts.device)
    return edit_costs, catastrophic_errors, minor_errors

def calc_tedn_scores(edit_costs, alpha_t=0.01):
    tedn_scores = torch.exp(-alpha_t * edit_costs)
    return tedn_scores

# uses outputs from calc_edit_costs
def calc_wellformedness(catastrophic_errors, minor_errors, gamma=3.0, alpha_w=0.2):
    wellformedness_scores = torch.zeros_like(catastrophic_errors, dtype=torch.float)
    # calculate well-formedness for non-catastrophic sequences
    wellformedness_scores = torch.exp(-alpha_w * minor_errors)
    # overwrite catastrophic sequences with penalty
    wellformedness_scores[catastrophic_errors] = -gamma
    return wellformedness_scores

def calc_token_f1(rollouts, target_lmx_seqs, pad_idx):
    # save original number of positions in each (ignoring padding) before truncation to use later
    num_predictions = (rollouts != pad_idx).sum(dim=-1)
    num_targets = (target_lmx_seqs != pad_idx).sum(dim=-1)

    # rollouts and target_lmx_seqs may have different max lengths, truncate longer one to calculate f1 at the positions we can
    max_rollout_len = rollouts.shape[-1]
    max_target_lmx_len = target_lmx_seqs.shape[-1]
    preds = rollouts[:, :min(max_rollout_len, max_target_lmx_len)]
    targets = target_lmx_seqs[:, :min(max_rollout_len, max_target_lmx_len)]

    # ignore positions where targets are padding
    true_positives = (preds == targets) & (targets != pad_idx)
    true_positives = true_positives.sum(dim=-1)
    precision = true_positives / (num_predictions + 1e-8)
    recall = true_positives / (num_targets + 1e-8)

    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    return f1_scores # (R, )

# non-overlapping n-grams since each token is a distinct unit and we want to catch multi-token loops
def calc_n_gram_penalty(rollouts, n, pad_idx):
    n_grams = rollouts.unfold(dimension=-1, size=n, step=n) # (R, num_grams, n)

    # right and left shift n-grams to align adjacent windows
    prev_n_grams = n_grams[:, :-1, :]
    next_n_grams = n_grams[:, 1:, :]
    pad_mask = torch.any((next_n_grams == pad_idx), dim=-1) # (R, num_grams), True = n-gram has pad_idx somewhere in it, False = n-gram has no pad_idx
    repeats = torch.all((prev_n_grams == next_n_grams), dim=-1) & ~pad_mask
    num_repeats = repeats.sum(dim=-1)
    repeat_opportunities = (~pad_mask).sum(dim=-1) # again, ignoring any n-grams with padding
    penalty = num_repeats / (repeat_opportunities + 1e-8)
    return penalty # (R, )

# not worth/probably worse to use torch.multiprocessing here over all values of n. Each call is too lightweight
def calc_repeat_penalty(rollouts, pad_idx, n_values=[1, 2, 3, 4]):
    total_penalty = 0
    for n in n_values:
        total_penalty += calc_n_gram_penalty(rollouts, n, pad_idx)
    repeat_penalty = total_penalty / len(n_values)
    return repeat_penalty # (R, )

def calc_len_penalty(rollout_mask, target_lmx_seqs, pad_idx, delta=5, tau=100):
    rollout_lens = rollout_mask.sum(dim=-1)
    target_lens = (target_lmx_seqs != pad_idx).sum(dim=-1)
    len_diffs = torch.abs(rollout_lens - target_lens)
    len_diffs[(len_diffs < delta)] = 0 # 0 penalty for differences within threshold
    penalty = torch.exp((torch.log(torch.tensor(2)) / tau) * len_diffs) - 1
    penalty = torch.clip(penalty, max=1.0)
    return penalty

# given component terms, calculate reward for each rollout then split into view of groups for normalization
def reward_rollouts(reward_config: RewardConfig, reward_components: RewardComponents, batch_size, num_rollouts_per_img):
    rewards = reward_config.lambda_tedn * reward_components.tedn_scores + reward_config.lambda_well_formed * reward_components.wellformedness_scores + reward_config.lambda_f1 * reward_components.f1_scores - reward_config.lambda_repeat * reward_components.repeat_penalty - reward_config.lambda_len * reward_components.len_penalty
    rewards = rewards.view(batch_size, num_rollouts_per_img)
    return rewards

# have to do a lot of logic here to deal with ragged rollouts
def calc_main_grpo_objective(theta_logits, rollouts, rollout_attention_mask, old_policy_log_probs, advantages, epsilon, num_groups):
    theta_log_probs = F.log_softmax(theta_logits, dim=-1)
    # get log probs for each token chosen in rollouts (besides <bos>) according to policy_theta
    left_shifted_rollouts = rollouts[:, 1:]
    theta_log_probs = torch.gather(theta_log_probs, dim=-1, index=left_shifted_rollouts.unsqueeze(-1))
    theta_log_probs = theta_log_probs.squeeze(-1) # remove dimension added for dimensions to match (which gather needs)
    # left shift old_policy to align with next token distributions from policy_theta
    old_policy_log_probs = old_policy_log_probs[:, 1:]
    ratios = torch.exp(theta_log_probs - old_policy_log_probs)
    unclipped = ratios * advantages.unsqueeze(1)
    clipped = torch.clip(ratios, min=(1 - epsilon), max=(1 + epsilon)) * advantages.unsqueeze(1)
    # rollout_attention_mask works here since it marks every position to make a prediction at
    unclipped[rollout_attention_mask] = 0
    clipped[rollout_attention_mask] = 0
    lens = (~rollout_attention_mask).sum(dim=-1)
    print(ratios)
    print(lens)

    grpo_objective = torch.min(unclipped, clipped)
    # average over each rollout
    grpo_objective = torch.sum(grpo_objective, dim=-1) / lens
    # average over groups
    grpo_objective = torch.sum(grpo_objective) / num_groups
    return grpo_objective

"""
GRPO update function for one minibatch
Inputs
    old_policy: reference policy to run rollouts with
    policy_theta: the policy to be updated
    batch: list of (image tensor, target_lmx_seq, target_musicxml_str)
    loss_config: LossConfig instance containing configuration for all the different loss/reward function weights    
    num_rollouts_per_img: number of rollouts to run on each input image
    grpo_epochs: number of GRPO update epochs to run on policy_theta
    device: device to run loop on
"""
def grpo_update(old_policy: GRPOViTOMR, policy_theta: GRPOViTOMR, batch: list[tuple[torch.Tensor, torch.Tensor, str]], rollout_config: RolloutConfig, reward_config: RewardConfig, loss_config: LossConfig, grpo_epochs, epsilon, device):
    pad_idx = old_policy.decoder.pad_idx
    imgs, target_lmx_seqs, target_musicxml_strs = zip(*batch)
    num_rollouts_per_img = rollout_config.num_rollouts_per_img
    batch_size = len(batch) # also equivalent to number of groups

    # rollout using old policy
    with torch.no_grad():
        with autocast(device_type=device, dtype=torch.bfloat16):
            # encode images into latent representations
            img_latent, latent_attention_mask = old_policy.encoder(imgs)
            # run rollouts
            img_latent, latent_attention_mask = old_policy.expand_img_latent_for_rollout(img_latent, latent_attention_mask, num_rollouts_per_img)
            rollouts, old_policy_log_probs, rollout_mask = old_policy.rollout_policy(img_latent, latent_attention_mask, max_actions=rollout_config.max_actions, top_k=rollout_config.top_k, temperature=rollout_config.temperature)

    # explicitly set tokens that aren't part of rollouts to <pad>. Later functions assume this is the case
    rollouts[~rollout_mask] = pad_idx

    # reward rollouts and calculate group-normalized advantages
    target_lmx_seqs = expand_target_lmx_seqs(target_lmx_seqs, num_rollouts_per_img, pad_idx, device)

    edit_costs, catastrophic_errors, minor_errors = calc_edit_costs(rollouts, pad_idx, batch_size, target_musicxml_strs, old_policy.decoder.idxs_to_tokens)
    tedn_scores = calc_tedn_scores(edit_costs, alpha_t=reward_config.alpha_tedn)
    wellformdness_scores = calc_wellformedness(catastrophic_errors, minor_errors, gamma=reward_config.gamma, alpha_w=reward_config.alpha_well_formed)
    f1_scores = calc_token_f1(rollouts, target_lmx_seqs, pad_idx)
    repeat_penalty = calc_repeat_penalty(rollouts, pad_idx)
    len_penalty = calc_len_penalty(rollout_mask, target_lmx_seqs, pad_idx, delta=reward_config.delta, tau=reward_config.tau)
    reward_components = RewardComponents(tedn_scores, wellformdness_scores, f1_scores, repeat_penalty, len_penalty)
    
    group_rewards = reward_rollouts(reward_config, reward_components, batch_size, num_rollouts_per_img) # (B, group_size)
    group_advantages = (group_rewards - group_rewards.mean(dim=-1, keepdim=True)) / (group_rewards.std(dim=-1, keepdim=True) + 1e-8)
    advantages = group_advantages.view(-1) # flatten to align with rolled out groups

    # policy update steps
    right_shifted_rollouts, rollout_attention_mask = old_policy.prepare_rollouts_for_policy_theta(rollouts, rollout_mask)
    for _ in range(grpo_epochs):
        # generate next token logits at each time step by using rollouts in a teacher forcing step
        theta_logits = policy_theta.decoder(right_shifted_rollouts, img_latent, rollout_attention_mask, latent_attention_mask)
        main_grpo_objective = calc_main_grpo_objective(theta_logits, rollouts, rollout_attention_mask, old_policy_log_probs, advantages, epsilon, batch_size)

    # calc a positive objective then negate it for loss

if __name__ == "__main__":
    # DEBUG
    from acai_omr.models.models import FineTuneOMREncoder, OMRDecoder, TeacherForcedViTOMR
    from acai_omr.train.omr_teacher_force_train import PE_MAX_HEIGHT, PE_MAX_WIDTH, MAX_LMX_SEQ_LEN
    from acai_omr.config import DEBUG_PRETRAINED_MAE_PATH   
    TEACHER_FORCED_STATE_DICT_PATH = "debug_teacher_forced_omr_train/debug_vitomr.pth"
    debug_kwargs = {"num_layers": 2, "num_heads": 1, "hidden_dim": 10, "mlp_dim": 1}
    pretrained_debug_encoder = FineTuneOMREncoder(16, PE_MAX_HEIGHT, PE_MAX_WIDTH, 1, **debug_kwargs)
    debug_decoder = OMRDecoder(MAX_LMX_SEQ_LEN, "lmx_vocab.txt", **debug_kwargs)
    debug_mae_state_dict = torch.load(DEBUG_PRETRAINED_MAE_PATH)
    debug_teacher_forced_vitomr = TeacherForcedViTOMR(pretrained_debug_encoder, debug_mae_state_dict, debug_decoder)

    debug_teacher_forced_state_dict = torch.load(TEACHER_FORCED_STATE_DICT_PATH)

    vitomr = GRPOViTOMR(pretrained_debug_encoder, debug_teacher_forced_vitomr.transition_head, debug_decoder, debug_teacher_forced_state_dict)

    teacher_forced_vitomr, base_img_transform, base_lmx_transform, device = set_up_omr_teacher_force_train()
    encoder = teacher_forced_vitomr.encoder
    transition_head = teacher_forced_vitomr.transition_head
    decoder = teacher_forced_vitomr.decoder

    teacher_forced_state_dict = torch.load(TEACHER_FORCED_STATE_DICT_PATH)

    # vitomr = GRPOViTOMR(encoder, transition_head, decoder, teacher_forced_state_dict)
    print(vitomr)

    # RL is more unstable and teacher force train already included augmentations, so slightly decrease strength
    camera_augment = v2.RandomApply(transforms=[
        v2.GaussianBlur(kernel_size=15, sigma=(0.2, 0.5)),
        v2.GaussianNoise(sigma=0.01),
        v2.RandomRotation(degrees=(-2, 2), interpolation=InterpolationMode.BILINEAR),
        v2.RandomPerspective(distortion_scale=0.2, p=1),
        v2.ColorJitter(brightness=0.1, saturation=0.2, contrast=0.2, hue=0),
    ], p=AUGMENTATION_P)

    grandstaff_camera_augment = v2.Compose([
        v2.RandomPerspective(distortion_scale=0.2, p=1),
        v2.ColorJitter(brightness=0.1, saturation=0.2, contrast=0.2, hue=0),
    ])

    olimpic_img_transform = v2.Compose([base_img_transform, camera_augment])

    grand_staff = GrandStaffLMXDataset(GRAND_STAFF_ROOT_DIR, "samples.train.txt", img_transform=base_img_transform, lmx_transform=base_lmx_transform, include_musicxml=True)
    olimpic = OlimpicDataset(OLIMPIC_SYNTHETIC_ROOT_DIR, "samples.train.txt", img_transform=olimpic_img_transform, lmx_transform=base_lmx_transform, include_musicxml=True)

    train_dataset = ConcatDataset([
        GrandStaffOMRTrainWrapper(grand_staff, AUGMENTATION_P, transform=grandstaff_camera_augment),
        olimpic,
    ])

    grand_staff_validate = GrandStaffLMXDataset(GRAND_STAFF_ROOT_DIR, "samples.dev.txt", img_transform=base_img_transform, lmx_transform=base_lmx_transform, include_musicxml=True)
    olimpic_synthetic_validate = OlimpicDataset(OLIMPIC_SYNTHETIC_ROOT_DIR, "samples.dev.txt", img_transform=base_img_transform, lmx_transform=base_lmx_transform, include_musicxml=True)
    olimpic_scanned_validate = OlimpicDataset(OLIMPIC_SCANNED_ROOT_DIR, "samples.dev.txt", img_transform=base_img_transform, lmx_transform=base_lmx_transform, include_musicxml=True)

    validation_dataset = ConcatDataset([
        GrandStaffOMRTrainWrapper(grand_staff_validate),
        olimpic_synthetic_validate,
        olimpic_scanned_validate,
    ])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=ragged_collate_fn, pin_memory=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=ragged_collate_fn, pin_memory=True)

    old_policy = copy.deepcopy(vitomr).eval()
    rollout_config = RolloutConfig(NUM_ROLLOUTS_PER_IMG, 10, 50, 1.2)
    reward_config = RewardConfig(5, 2, 3, 2, 0.25, 0.01, 0.2, 3.0, 5, 100)
    loss_config = LossConfig(0.2, 0.1)
    print(f"Rollout hyperparameters: {rollout_config}")
    print(f"Reward hyperparameters: {reward_config}")
    print(f"Loss hyperparameters: {loss_config}")
    for batch in train_dataloader:
        grpo_update(old_policy, vitomr, batch, rollout_config, reward_config, loss_config, NUM_ROLLOUTS_PER_IMG, GRPO_EPOCHS, device)
        exit()

# things to track: magnitude of each reward component, each loss component, curriculum parts (things that are being increased/annealed over time)
