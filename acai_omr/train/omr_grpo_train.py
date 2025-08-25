import torch
from torch.amp import autocast
from acai_omr.train.omr_teacher_force_train import set_up_omr_teacher_force_train
from acai_omr.models.models import GRPOViTOMR
from acai_omr.utils.utils import stringify_lmx_seq
from olimpic_app.evaluation.TEDn_lmx_xml import TEDn_lmx_xml
from dataclasses import dataclass
from multiprocessing import Pool

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

# target_musicxml_strs is a tuple where target_musicxml_strs[i] = the target musicxml for image i in the minibatch
# Returns three tuples of metrics across all rollouts, one for raw edit_costs, one for a bool on if a castrophic error 
# happened, and one for counts of minor deliniearization errors
def calc_edit_costs(rollouts, pad_idx, batch_size, num_rollouts_per_img, target_musicxml_strs, idxs_to_tokens, num_parallel_processes=12):
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
    return edit_costs, catastrophic_errors, minor_errors

def calc_tedn_scores(edit_costs, alpha_t=0.01):
    tedn_scores = torch.exp(-alpha_t * edit_costs)
    return tedn_scores

def calc_wellformedness(catastrophic_errors, minor_errors, device, gamma=3.0, alpha_w=0.2):
    catastrophic_errors = torch.tensor(catastrophic_errors, device=device)
    minor_errors = torch.tensor(minor_errors, device=device)
    wellformedness_scores = torch.zeros_like(catastrophic_errors, dtype=torch.float)
    # calculate well-formedness for non-catastrophic sequences
    wellformedness_scores = torch.exp(-alpha_w * minor_errors)
    # overwrite catastrophic sequences with penalty
    wellformedness_scores[catastrophic_errors] = -gamma
    return wellformedness_scores

# broadcast sequences across rollouts into an (R x T) padded tensor
def expand_target_lmx_seqs(target_lmx_seqs: tuple[torch.Tensor], num_rollouts_per_img, pad_idx, device):
    target_lmx_seqs = torch.nested.as_nested_tensor(target_lmx_seqs, device=device, layout=torch.jagged)
    target_lmx_seqs = target_lmx_seqs.to_padded_tensor(padding=pad_idx)
    target_lmx_seqs = target_lmx_seqs.unsqueeze(1)
    target_lmx_seqs = target_lmx_seqs.expand(-1, num_rollouts_per_img, -1)
    target_lmx_seqs = target_lmx_seqs.flatten(start_dim=0, end_dim=1)
    return target_lmx_seqs

def calc_token_f1(rollouts, target_lmx_seqs, pad_idx):
    # save original number of positions in each (ignoring padding) before truncation to use later
    num_predictions = (rollouts != pad_idx).sum().item()
    num_targets = (target_lmx_seqs != pad_idx).sum().item()

    # rollouts and target_lmx_seqs may have different max lengths
    max_rollout_len = rollouts.shape[-1]
    max_target_lmx_len = target_lmx_seqs.shape[-1]
    # truncate longer one to calculate f1 at the positions we can
    
    rollouts = rollouts[:, :min(max_rollout_len, max_target_lmx_len)]
    target_lmx_seqs = target_lmx_seqs[:, :min(max_rollout_len, max_target_lmx_len)]

    rollouts = rollouts.flatten(-1)
    target_lmx_seqs = target_lmx_seqs.flatten(-1)

    pad_mask = target_lmx_seqs != pad_idx # ignore padding in target tensor
    preds = rollouts[pad_mask]
    targets = target_lmx_seqs[pad_mask]

    true_positives = (preds == targets).sum().item()
    precision = true_positives / num_predictions
    recall = true_positives / num_targets

    f1_score = 2 * precision * recall / (precision + recall + 1e-8)
    return f1_score

# def all the penalties/bonuses

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
    imgs, target_lmx_seqs, target_musicxml_strs = zip(*batch)

    # rollout using old policy
    with torch.no_grad():
        with autocast(device_type=device, dtype=torch.bfloat16):
            # encode images into latent representations
            img_latent, latent_attention_mask = old_policy.encoder(imgs)
            # run rollouts
            img_latent, latent_attention_mask = old_policy.expand_img_latent_for_rollout(img_latent, latent_attention_mask, num_rollouts_per_img)
            rollouts, old_policy_log_probs, rollout_mask = old_policy.rollout_policy(img_latent, latent_attention_mask)

    # reward rollouts and calculate group-normalized advantages
    pad_idx = old_policy.decoder.pad_idx
    target_lmx_seqs = expand_target_lmx_seqs(target_lmx_seqs, num_rollouts_per_img, pad_idx, device)

    # policy update steps
    rollouts, rollout_attention_mask, rollout_lens = old_policy.prepare_rollouts_for_policy_theta(rollouts, rollout_mask)
    for i in range(grpo_epochs):
        # generate next token logits at each time step through treating rollouts like a teacher forcing step
        logits = policy_theta.decoder(rollouts, img_latent, rollout_attention_mask, latent_attention_mask)

if __name__ == "__main__":
    TEACHER_FORCED_STATE_DICT_PATH = "debug_teacher_forced_omr_train/debug_vitomr.pth"

    teacher_forced_vitomr, base_img_transform, base_lmx_transform, device = set_up_omr_teacher_force_train()
    encoder = teacher_forced_vitomr.encoder
    transition_head = teacher_forced_vitomr.transition_head
    decoder = teacher_forced_vitomr.decoder

    teacher_forced_state_dict = torch.load(TEACHER_FORCED_STATE_DICT_PATH)

    vitomr = GRPOViTOMR(encoder, transition_head, decoder, teacher_forced_state_dict)
    print(vitomr)
