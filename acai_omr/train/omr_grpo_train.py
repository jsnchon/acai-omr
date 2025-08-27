import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.multiprocessing import Pool # regular multiprocessing complains about torch's multi-threadedness
from acai_omr.train.omr_teacher_force_train import set_up_omr_teacher_force_train
from acai_omr.train.datasets import GrandStaffLMXDataset, GrandStaffOMRTrainWrapper, OlimpicDataset 
from acai_omr.models.models import GRPOViTOMR, OMRCELoss
from acai_omr.utils.utils import stringify_lmx_seq, ragged_collate_fn, GRPOLogger
from acai_omr.config import GRAND_STAFF_ROOT_DIR, OLIMPIC_SYNTHETIC_ROOT_DIR, OLIMPIC_SCANNED_ROOT_DIR
from torchvision.transforms import v2, InterpolationMode
from olimpic_app.evaluation.TEDn_lmx_xml import TEDn_lmx_xml
from dataclasses import dataclass
import copy
import time

@dataclass
class RolloutConfig:
    group_size: int
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

    def get_configs(self):
        return rollout_config, reward_config, loss_config, update_config

LOG_DIR = "runs/grpo_train"

AUGMENTATION_P = 0.3

BATCH_SIZE = 16
NUM_WORKERS = 6

LR = 5e-5
ADAMW_BETAS = (0.9, 0.999)
ADAMW_WEIGHT_DECAY = 0.01

# all in outer steps (ie minibatches)
EPOCHS = 8
EVAL_FREQUENCY = 500
CHECKPOINT_FREQUENCY = 500

INITIAL_ROLLOUT_CONFIG = RolloutConfig(
    group_size=8, 
    max_actions=10, 
    top_k=50, 
    temperature=1.2
)
INITIAL_REWARD_CONFIG = RewardConfig(
    lambda_tedn=5,
    lambda_well_formed=2,
    lambda_f1=3,
    lambda_repeat=2,
    lambda_len=0.25,
    alpha_tedn=0.01,
    alpha_well_formed=0.2,
    gamma=3,
    delta=5,
    tau=100
)
INITIAL_LOSS_CONFIG = LossConfig(
    entropy_beta=0.3,
    lambda_ce=0.15
)
INITIAL_UPDATE_CONFIG = UpdateConfig(
    epsilon=0.2,
    update_epochs=2,
    max_grad_norm=1.0
)

TEACHER_FORCED_STATE_DICT_PATH = ""

# broadcast sequences across rollouts into an (R x T) padded tensor
def expand_target_lmx_seqs(target_lmx_seqs: tuple[torch.Tensor], group_size, pad_idx, device):
    target_lmx_seqs = torch.nested.as_nested_tensor(target_lmx_seqs, device=device, layout=torch.jagged)
    target_lmx_seqs = target_lmx_seqs.to_padded_tensor(padding=pad_idx)
    target_lmx_seqs = target_lmx_seqs.unsqueeze(1)
    target_lmx_seqs = target_lmx_seqs.expand(-1, group_size, -1)
    target_lmx_seqs = target_lmx_seqs.flatten(start_dim=0, end_dim=1)
    return target_lmx_seqs

# all these reward components return (R, ) tensors, one component per rollout each

# target_musicxml_strs is a tuple where target_musicxml_strs[i] = the target musicxml for image i in the minibatch
# Returns three tensors of metrics across all rollouts, one for raw edit_costs, one for a bool on if a castrophic error 
# happened, and one for counts of minor delinearization errors
def calc_edit_costs(rollouts, pad_idx, num_groups, group_size, target_musicxml_strs: tuple, idxs_to_tokens, num_parallel_processes=12):
    # unfold into individual groups since we can't broadcast the musicxml strings within groups
    rollout_groups = rollouts.view(num_groups, group_size, rollouts.shape[-1])
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
    wellformedness_scores = wellformedness_scores.masked_fill(catastrophic_errors, -gamma)
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
    len_diffs = len_diffs.masked_fill((len_diffs < delta), 0) # 0 penalty for differences within threshold
    penalty = torch.exp((torch.log(torch.tensor(2)) / tau) * len_diffs) - 1
    penalty = torch.clip(penalty, max=1.0)
    return penalty

# given component terms, calculate reward for each rollout then split into view of groups for normalization
def reward_rollouts(reward_config: RewardConfig, reward_components: RewardComponents, num_groups, group_size):
    rewards = reward_config.lambda_tedn * reward_components.tedn_scores + reward_config.lambda_well_formed * reward_components.wellformedness_scores + reward_config.lambda_f1 * reward_components.f1_scores - reward_config.lambda_repeat * reward_components.repeat_penalty - reward_config.lambda_len * reward_components.len_penalty
    rewards = rewards.view(num_groups, group_size)
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
    unclipped = unclipped.masked_fill(rollout_attention_mask, 0)
    clipped = clipped.masked_fill(rollout_attention_mask, 0)
    lens = (~rollout_attention_mask).sum(dim=-1)

    grpo_objective = torch.min(unclipped, clipped)
    # average over each rollout (note rollout and group averages have to be separate; can't just do
    # one torch.mean() due to ragged masked filling of tensors)
    grpo_objective = torch.sum(grpo_objective, dim=-1) / lens
    # average over groups (not over all rollouts)
    grpo_objective = torch.sum(grpo_objective) / num_groups
    return grpo_objective

# returns policy_theta's average entropy per rollout
def calc_policy_theta_entropy(theta_logits, rollout_attention_mask):
    probs = F.softmax(theta_logits, dim=-1)
    log_probs = F.log_softmax(theta_logits, dim=-1)
    entropies = -probs * log_probs
    entropies = entropies.sum(dim=-1)
    # mask out invalid positions and average over rollouts
    entropies = entropies.masked_fill(rollout_attention_mask, 0)
    lens = (~rollout_attention_mask).sum(dim=-1)
    # get an average for each rollout
    entropies = entropies.sum(dim=-1) / lens # (R, )
    return entropies

def calc_entropy_bonus(theta_logits, rollout_attention_mask, vocab_size):
    rollout_entropies = calc_policy_theta_entropy(theta_logits, rollout_attention_mask)
    # average over all rollouts
    avg_entropies = rollout_entropies.mean()
    raw_bonus = avg_entropies / torch.log(torch.tensor(vocab_size)) # normalize to [0, 1]
    return raw_bonus

def calc_teacher_forced_ce_loss(policy_theta, unexpanded_img_latent, unexpanded_latent_attention_mask, unexpanded_target_lmx_seqs, ce_loss_fn):
    logits, target_seqs = policy_theta.forward_teacher_forced(unexpanded_img_latent, unexpanded_latent_attention_mask, unexpanded_target_lmx_seqs, checkpoint_grads=True)
    ce_loss = ce_loss_fn(logits, target_seqs)
    return ce_loss

"""
GRPO update function for one minibatch
Inputs
    old_policy: reference policy to run rollouts with
    policy_theta: the policy to be updated
    batch: list of (image tensor, target_lmx_seq, target_musicxml_str)
    loss_config: LossConfig instance containing configuration for all the different loss/reward function weights    
    group_size: number of rollouts to run on each input image
    grpo_epochs: number of GRPO update epochs to run on policy_theta
    device: device to run loop on
"""
def grpo_update(old_policy: GRPOViTOMR, policy_theta: GRPOViTOMR, optimizer, batch: list[tuple[torch.Tensor, torch.Tensor, str]], grpo_config: GRPOConfig, ce_loss_fn: OMRCELoss, device, logger: GRPOLogger, batch_step):
    rollout_config, reward_config, loss_config, update_config = grpo_config.get_configs()

    pad_idx = old_policy.decoder.pad_idx
    unexpanded_imgs, unexpanded_target_lmx_seqs, target_musicxml_strs = zip(*batch)
    group_size = rollout_config.group_size
    num_groups = len(batch) # batch_size = num_groups
    vocab_size = policy_theta.decoder.vocab_embedding.num_embeddings

    # rollout using old policy
    with torch.no_grad():
        with autocast(device_type=device, dtype=torch.bfloat16):
            # encode images into latent representations
            unexpanded_img_latent, unexpanded_latent_attention_mask = old_policy.encoder(unexpanded_imgs)
            # run rollouts
            img_latent, latent_attention_mask = old_policy.expand_img_latent_for_rollout(unexpanded_img_latent, unexpanded_latent_attention_mask, group_size)
            rollouts, old_policy_log_probs, rollout_mask = old_policy.forward_rollout_policy(img_latent, latent_attention_mask, max_actions=rollout_config.max_actions, top_k=rollout_config.top_k, temperature=rollout_config.temperature)

    # explicitly set tokens that aren't part of rollouts to <pad>. Later functions assume this is the case
    rollouts = rollouts.masked_fill(~rollout_mask, pad_idx)

    # reward rollouts and calculate group-normalized advantages. Note these targets aren't left-shifted
    target_lmx_seqs = expand_target_lmx_seqs(unexpanded_target_lmx_seqs, group_size, pad_idx, device)

    edit_costs, catastrophic_errors, minor_errors = calc_edit_costs(rollouts, pad_idx, num_groups, group_size, target_musicxml_strs, old_policy.decoder.idxs_to_tokens)
    tedn_scores = calc_tedn_scores(edit_costs, alpha_t=reward_config.alpha_tedn)
    wellformedness_scores = calc_wellformedness(catastrophic_errors, minor_errors, gamma=reward_config.gamma, alpha_w=reward_config.alpha_well_formed)
    f1_scores = calc_token_f1(rollouts, target_lmx_seqs, pad_idx)
    repeat_penalty = calc_repeat_penalty(rollouts, pad_idx)
    len_penalty = calc_len_penalty(rollout_mask, target_lmx_seqs, pad_idx, delta=reward_config.delta, tau=reward_config.tau)
    reward_components = RewardComponents(tedn_scores, wellformedness_scores, f1_scores, repeat_penalty, len_penalty)
    
    group_rewards = reward_rollouts(reward_config, reward_components, num_groups, group_size) # (B, group_size)
    group_advantages = (group_rewards - group_rewards.mean(dim=-1, keepdim=True)) / (group_rewards.std(dim=-1, keepdim=True) + 1e-8)
    advantages = group_advantages.view(-1) # flatten to align with rolled out groups

    logger.log_group_rewards(group_rewards, batch_step)
    logger.log_group_advantages(group_advantages, batch_step)

    # policy update steps
    right_shifted_rollouts, rollout_attention_mask = old_policy.prepare_rollouts_for_policy_theta(rollouts, rollout_mask)
    for update_epoch in range(update_config.update_epochs):
        # generate next token logits at each time step by using rollouts in a teacher forcing step
        with autocast(device_type=device, dtype=torch.bfloat16):
            theta_logits = policy_theta.decoder(right_shifted_rollouts, img_latent, rollout_attention_mask, latent_attention_mask, checkpoint_grads=True)
            main_grpo_objective = calc_main_grpo_objective(theta_logits, rollouts, rollout_attention_mask, old_policy_log_probs, advantages, update_config.epsilon, num_groups)
            entropy_bonus = calc_entropy_bonus(theta_logits, rollout_attention_mask, vocab_size)
            logger.log_raw_reward_components(reward_components, entropy_bonus)
            reward = main_grpo_objective + loss_config.entropy_beta * entropy_bonus
            logger.log_reward(reward, batch_step, update_epoch)

            if loss_config.lambda_ce:
                ce_loss = calc_teacher_forced_ce_loss(policy_theta, unexpanded_img_latent, unexpanded_latent_attention_mask, unexpanded_target_lmx_seqs, ce_loss_fn)
            else:
                ce_loss = 0

            loss = -reward + loss_config.lambda_ce * ce_loss
            logger.log_loss(loss, ce_loss, batch_step, update_epoch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_theta.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

def epoch_loop(old_policy, policy_theta, optimizer, grpo_config, ce_loss_fn, logger, device, epoch_num):
    for batch_step, batch in enumerate(train_dataloader):
        grpo_config = GRPOConfig(rollout_config, reward_config, loss_config, update_config)
        logger.log_configs(grpo_config, batch_step)
        # each batch, we need to refresh old_policy to match the resulting updated policy_theta from the previous GRPO updates.
        # Instead of repeatedly deepcopying, and since the encoder and transition head are frozen, we can just refresh the decoder parameters
        old_policy.decoder.load_state_dict(policy_theta.decoder.state_dict())
        grpo_update(old_policy, policy_theta, optimizer, batch, grpo_config, ce_loss_fn, device, logger, batch_step)
        # schedule step configs
        if batch_step == 5:
            exit()

        # check overall step, stop/checkpoint if need to


# def validation_loop()

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

    policy_theta = GRPOViTOMR(pretrained_debug_encoder, debug_teacher_forced_vitomr.transition_head, debug_decoder, debug_teacher_forced_state_dict)
    # END DEBUG

    teacher_forced_vitomr, base_img_transform, base_lmx_transform, device = set_up_omr_teacher_force_train()
    encoder = teacher_forced_vitomr.encoder
    transition_head = teacher_forced_vitomr.transition_head
    decoder = teacher_forced_vitomr.decoder

    teacher_forced_state_dict = torch.load(TEACHER_FORCED_STATE_DICT_PATH)

    # policy_theta = GRPOViTOMR(encoder, transition_head, decoder, teacher_forced_state_dict)
    print(f"{policy_theta}\n")

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

    old_policy = copy.deepcopy(policy_theta).eval()
    for p in old_policy.parameters():
        p.requires_grad = False
    # TODO: turn all these hyperparams into constants (or the configs themselves into constants)
    rollout_config = INITIAL_ROLLOUT_CONFIG
    reward_config = INITIAL_REWARD_CONFIG
    loss_config = INITIAL_LOSS_CONFIG
    update_config = INITIAL_UPDATE_CONFIG
    print(f"Initial GRPO hyperparameters\n{'-' * 50}\nRollout hyperparameters: {rollout_config}\nReward hyperparameters: {reward_config}\nLoss hyperparameters: {loss_config}\nGRPO update hyperparameters: {update_config}")
    ce_loss_fn = OMRCELoss(pad_idx=old_policy.decoder.pad_idx, label_smoothing=0.0)

    optimizer = torch.optim.AdamW(policy_theta.parameters(), )

    writer = SummaryWriter(log_dir=LOG_DIR, max_queue=300) # flushes around every 20 minibatches (~15 logs per minibatch)
    logger = GRPOLogger(writer, update_config.update_epochs)

    # TRAIN LOOP START (TODO: decompose into its own function?)

    # needs initial GRPOconfig, logger, old policy, policy theta, optimizer, ce loss fn, device, epoch num

    for epoch_num in range(EPOCHS):
        epoch_start_time = time.perf_counter()
        epoch_loop(old_policy, policy_theta, optimizer, grpo_config, ce_loss_fn, logger, device, epoch_num)
        epoch_end_time = time.perf_counter()
# things to track: magnitude of each reward component, each loss component, curriculum parts (things that are being increased/annealed over time)
