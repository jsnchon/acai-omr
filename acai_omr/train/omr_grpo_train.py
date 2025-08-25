import torch
import multiprocessing as mp
from torch.amp import autocast
from acai_omr.train.omr_teacher_force_train import set_up_omr_teacher_force_train
from acai_omr.models.models import GRPOViTOMR
from acai_omr.utils.utils import stringify_lmx_seq
from olimpic_app.evaluation.TEDn_lmx_xml import TEDn_lmx_xml
from dataclasses import dataclass
from multiprocessing import Pool

mp.set_start_method("forkserver") # PyTorch is multi-threaded, so "fork" (the default) is dangerous

TEACHER_FORCED_STATE_DICT_PATH = ""
# make sure to set include_musicxml = True for the datasets

@dataclass
class RewardLambdas:
    lambda_tedn: float
    lambda_well_formed: float
    lambda_f1: float
    lambda_repeat: float
    lambda_len: float

@dataclass
class LossConfig:
    reward_lambdas: RewardLambdas
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

    f1_score = 2 * precision * recall / (precision + recall + 1e-8)
    return f1_score # (R, )

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
    overall_penalty = total_penalty / len(n_values)
    return overall_penalty # (R, )

def calc_len_penalty(rollout_mask, target_lmx_seqs, pad_idx, delta=5, tau=100):
    rollout_lens = rollout_mask.sum(dim=-1)
    target_lens = (target_lmx_seqs != pad_idx).sum(dim=-1)
    len_diffs = torch.abs(rollout_lens - target_lens)
    len_diffs[(len_diffs < delta)] = 0 # 0 penalty for differences within threshold
    penalty = torch.exp((torch.log(torch.tensor(2)) / tau) * len_diffs) - 1
    penalty = torch.clip(penalty, max=1.0)
    return penalty

# normalize rewards within each rollout group
# def group_normalize_rewards()

# calculate reward for each rollout
# def reward_sequences()


"""
GRPO update function for one minibatch
Inputs
    old_policy: reference policy to run rollouts with
    policy_theta: the policy to be updated
    batch: list of (image tensor, target_lmx_seq, target_musicxml_str)
    loss_config: LossConfig instance containing configuration for all the different loss/reward function weights    
    device: device to run loop on
    num_rollouts_per_img: number of rollouts to run on each input image
    grpo_epochs: number of GRPO update epochs to run on policy_theta
"""
def grpo_update(old_policy: GRPOViTOMR, policy_theta: GRPOViTOMR, batch: list[tuple[torch.Tensor, torch.Tensor, str]], loss_config: LossConfig, device, num_rollouts_per_img=4, grpo_epochs=6):
    pad_idx = old_policy.decoder.pad_idx
    imgs, target_lmx_seqs, target_musicxml_strs = zip(*batch)

    # rollout using old policy
    with torch.no_grad():
        with autocast(device_type=device, dtype=torch.bfloat16):
            # encode images into latent representations
            img_latent, latent_attention_mask = old_policy.encoder(imgs)
            # run rollouts
            img_latent, latent_attention_mask = old_policy.expand_img_latent_for_rollout(img_latent, latent_attention_mask, num_rollouts_per_img)
            rollouts, old_policy_log_probs, rollout_mask = old_policy.rollout_policy(img_latent, latent_attention_mask)

    # explicitly set tokens that aren't part of rollouts to <pad>. Later functions assume this is the case
    rollouts[~rollout_mask] = pad_idx

    # reward rollouts and calculate group-normalized advantages
    target_lmx_seqs = expand_target_lmx_seqs(target_lmx_seqs, num_rollouts_per_img, pad_idx, device)

    # policy update steps
    right_shifted_rollouts, rollout_attention_mask = old_policy.prepare_rollouts_for_policy_theta(rollouts, rollout_mask)
    for i in range(grpo_epochs):
        # generate next token logits at each time step by using rollouts in a teacher forcing step
        logits = policy_theta.decoder(right_shifted_rollouts, img_latent, rollout_attention_mask, latent_attention_mask)

if __name__ == "__main__":
    TEACHER_FORCED_STATE_DICT_PATH = "debug_teacher_forced_omr_train/debug_vitomr.pth"

    teacher_forced_vitomr, base_img_transform, base_lmx_transform, device = set_up_omr_teacher_force_train()
    encoder = teacher_forced_vitomr.encoder
    transition_head = teacher_forced_vitomr.transition_head
    decoder = teacher_forced_vitomr.decoder

    teacher_forced_state_dict = torch.load(TEACHER_FORCED_STATE_DICT_PATH)

    vitomr = GRPOViTOMR(encoder, transition_head, decoder, teacher_forced_state_dict)
    print(vitomr)
