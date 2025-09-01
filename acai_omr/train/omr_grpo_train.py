import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.multiprocessing import Pool # regular multiprocessing complains about torch's multi-threadedness
from acai_omr.train.omr_teacher_force_train import set_up_omr_teacher_force_train
from acai_omr.train.datasets import GrandStaffLMXDataset, GrandStaffOMRTrainWrapper, OlimpicDataset
from acai_omr.models.models import GRPOViTOMR, OMRCELoss, OMRDecoder
from acai_omr.utils.utils import stringify_lmx_seq, ragged_collate_fn, stepwise_cosine_anneal_with_warmup
from acai_omr.utils.utils import RolloutConfig, RewardConfig, RewardComponents, LossConfig, UpdateConfig, GRPOConfig, StepCounter, GRPOLogger
from acai_omr.config import GRAND_STAFF_ROOT_DIR, OLIMPIC_SYNTHETIC_ROOT_DIR, OLIMPIC_SCANNED_ROOT_DIR
from torchvision.transforms import v2, InterpolationMode
from olimpic_app.evaluation.TEDn_lmx_xml import TEDn_lmx_xml
from pathlib import Path
import pandas as pd
import copy
import time

MODEL_DIR_PATH = Path("grpo_omr_train")
CHECKPOINTS_DIR_PATH = MODEL_DIR_PATH / "checkpoints"
LOG_DIR = "runs/grpo"

AUGMENTATION_P = 0.25

TRAIN_BATCH_SIZE = 16 
VALIDATION_BATCH_SIZE = 128
NUM_WORKERS = 26

LR = 2e-5
ADAMW_BETAS = (0.9, 0.999)
ADAMW_WEIGHT_DECAY = 0.01

EPOCHS = 8 # enough for ~25000 outer steps

MINI_VALIDATION_SIZE = 1000 # subset size

# in outer steps within each epoch (ie minibatches)
MINI_VALIDATION_FREQ = 250
CHECKPOINT_FREQ = 1000

TEACHER_FORCED_STATE_DICT_PATH = "tf_omr_train/vitomr.pth"

INITIAL_ROLLOUT_CONFIG = RolloutConfig(
    group_size=8, 
    max_actions=768, 
    top_k=50, 
    temperature=1.2
)
INITIAL_REWARD_CONFIG = RewardConfig(
    lambda_tedn=6,
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
    entropy_beta=0.25,
    lambda_ce=0.15
)
INITIAL_UPDATE_CONFIG = UpdateConfig(
    epsilon=0.2,
    update_epochs=2,
    max_grad_norm=1.0
)

# all these schedulers step per-batch
WARMUP_STEPS = 750 # 3% of total steps
MIN_LR = 4e-6

EXPLORATION_EPOCHS = 1
MAX_MAX_ACTIONS = 1536
MIN_TOP_K = 10
MIN_TEMPERATURE = 0.7
MAX_LAMBDA_LEN = 2 
MIN_ENTROPY_BETA = 0
MIN_LAMBDA_CE = 0

class CurriculumScheduler:
    def __init__(self, grpo_config, exploration_epochs, total_epochs, num_outer_steps_per_epoch,
                 max_max_actions, min_top_k, min_temperature, max_lambda_len, min_beta, min_lambda_ce):
        self.grpo_config = grpo_config
        self.step_count = 0
        self.exploration_steps = exploration_epochs * num_outer_steps_per_epoch
        self.anneal_steps = (total_epochs - exploration_epochs) * num_outer_steps_per_epoch
        self.total_steps = self.exploration_steps + self.anneal_steps
        # tuples of (init_value, min/max_value) depending on if being increased or annealed
        self.max_actions = (grpo_config.rollout_config.max_actions, max_max_actions)
        self.top_k = (grpo_config.rollout_config.top_k, min_top_k)
        self.temperature = (grpo_config.rollout_config.temperature, min_temperature)
        self.lambda_len = (grpo_config.reward_config.lambda_len, max_lambda_len)
        self.entropy_beta = (grpo_config.loss_config.entropy_beta, min_beta)
        self.lambda_ce = (grpo_config.loss_config.lambda_ce, min_lambda_ce)

    def calc_increasing_value(self, progress, init_value, max_value):
        return init_value + progress * (max_value - init_value)

    def calc_annealing_value(self, progress, init_value, min_value):
        return init_value - progress * (init_value - min_value)

    def step(self):
        if self.step_count < self.exploration_steps: # if still in the exploration stage, don't change hyperparameters
            self.step_count += 1
            return

        progress = (self.step_count - self.exploration_steps) / self.anneal_steps
        self.grpo_config.rollout_config.max_actions = int(self.calc_increasing_value(progress, *self.max_actions))
        self.grpo_config.rollout_config.top_k = int(self.calc_annealing_value(progress, *self.top_k))
        self.grpo_config.rollout_config.temperature = self.calc_annealing_value(progress, *self.temperature)
        self.grpo_config.reward_config.lambda_len = self.calc_increasing_value(progress, *self.lambda_len)
        self.grpo_config.loss_config.entropy_beta = self.calc_annealing_value(progress, *self.entropy_beta)
        self.grpo_config.loss_config.lambda_ce = self.calc_annealing_value(progress, *self.lambda_ce)
        
        self.step_count += 1 

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

def calc_len_penalty(rollout_mask, target_lmx_seqs, pad_idx, delta=10, tau=100):
    rollout_lens = rollout_mask.sum(dim=-1)
    target_lens = (target_lmx_seqs != pad_idx).sum(dim=-1)
    len_diffs = torch.abs(rollout_lens - target_lens)
    len_diffs = len_diffs.masked_fill((len_diffs < delta), 0) # 0 penalty for differences within threshold
    penalty = torch.exp((torch.log(torch.tensor(2)) / tau) * len_diffs) - 1
    penalty = torch.clip(penalty, max=1.0)
    return penalty

# given component terms, calculate reward for each rollout then split into view of groups for normalization
def calc_group_rewards(reward_config: RewardConfig, reward_components: RewardComponents, num_groups, group_size):
    rewards = reward_config.lambda_tedn * reward_components.tedn_scores + reward_config.lambda_well_formed * reward_components.wellformedness_scores + reward_config.lambda_f1 * reward_components.f1_scores - reward_config.lambda_repeat * reward_components.repeat_penalty - reward_config.lambda_len * reward_components.len_penalty
    rewards = rewards.view(num_groups, group_size)
    return rewards

def reward_rollouts(reward_config, rollouts, rollout_mask, target_lmx_seqs, target_musicxml_strs, num_groups, group_size, idxs_to_tokens, pad_idx):
    edit_costs, catastrophic_errors, minor_errors = calc_edit_costs(rollouts, pad_idx, num_groups, group_size, target_musicxml_strs, idxs_to_tokens)
    tedn_scores = calc_tedn_scores(edit_costs, alpha_t=reward_config.alpha_tedn)
    wellformedness_scores = calc_wellformedness(catastrophic_errors, minor_errors, gamma=reward_config.gamma, alpha_w=reward_config.alpha_well_formed)
    f1_scores = calc_token_f1(rollouts, target_lmx_seqs, pad_idx)
    repeat_penalty = calc_repeat_penalty(rollouts, pad_idx)
    len_penalty = calc_len_penalty(rollout_mask, target_lmx_seqs, pad_idx, delta=reward_config.delta, tau=reward_config.tau)
    reward_components = RewardComponents(tedn_scores, wellformedness_scores, f1_scores, repeat_penalty, len_penalty)
    
    raw_group_rewards = calc_group_rewards(reward_config, reward_components, num_groups, group_size) # (B, group_size)
    return raw_group_rewards, reward_components

# have to do a lot of logic here to deal with ragged rollouts
def calc_grpo_objective(theta_logits, rollouts, rollout_attention_mask, old_policy_log_probs, advantages, epsilon, num_groups):
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
    # average over each rollout (note averages over rollouts then groups have to be separate; can't just do
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
Outputs
    A tuple of metrics averaged across what makes sense. Loss is averaged over update epochs, raw reward is averaged across 
    all rollouts, and each component stored in reward_components is averaged over all rollouts
Naming notes
    Raw reward (or just reward) is the plain task performance signal we really care about
    GRPO objective is the main GRPO objective (clipped advantages from raw reward weighted by ratios)
    Shaped objective is the GRPO objective plus auxiliary terms like entropy bonus, CE loss
    Train loss is just the negative of the overall objective
"""
def grpo_update(old_policy: GRPOViTOMR, policy_theta: GRPOViTOMR, optimizer, batch: list[tuple[torch.Tensor, torch.Tensor, str]], grpo_config: GRPOConfig, ce_loss_fn: OMRCELoss, device, logger: GRPOLogger, counter):
    rollout_config, reward_config, loss_config, update_config = grpo_config.get_configs()

    pad_idx = old_policy.decoder.pad_idx
    unexpanded_imgs, unexpanded_target_lmx_seqs, target_musicxml_strs = zip(*batch)
    unexpanded_imgs = [unexpanded_img.to(device) for unexpanded_img in unexpanded_imgs]
    unexpanded_target_lmx_seqs = [unexpanded_target_lmx_seq.to(device) for unexpanded_target_lmx_seq in unexpanded_target_lmx_seqs]

    group_size = rollout_config.group_size
    num_groups = len(batch) # batch_size = num_groups
    vocab_size = policy_theta.decoder.vocab_embedding.num_embeddings

    # rollout using old policy
    with torch.no_grad():
        with autocast(device_type=device, dtype=torch.bfloat16):
            # encode images into latent representations
            unexpanded_img_latent, unexpanded_latent_attention_mask = old_policy.encoder(unexpanded_imgs)
            unexpanded_img_latent = old_policy.transition_head(unexpanded_img_latent)
            # run rollouts
            img_latent, latent_attention_mask = old_policy.expand_img_latent_for_rollout(unexpanded_img_latent, unexpanded_latent_attention_mask, group_size)
            rollouts, old_policy_log_probs, rollout_mask = old_policy.cached_forward_rollout_policy(img_latent, latent_attention_mask, max_actions=rollout_config.max_actions, top_k=rollout_config.top_k, temperature=rollout_config.temperature)

    # reward rollouts and calculate group-normalized advantages. Note these targets aren't left-shifted
    target_lmx_seqs = expand_target_lmx_seqs(unexpanded_target_lmx_seqs, group_size, pad_idx, device)
    raw_group_rewards, reward_components = reward_rollouts(reward_config, rollouts, rollout_mask, target_lmx_seqs, target_musicxml_strs, num_groups, group_size, old_policy.decoder.idxs_to_tokens, pad_idx)
    logger.log_raw_reward_stats(raw_group_rewards, counter.global_step)
    logger.log_raw_reward_components(reward_components, counter.global_step)

    group_advantages = (raw_group_rewards - raw_group_rewards.mean(dim=-1, keepdim=True)) / (raw_group_rewards.std(dim=-1, keepdim=True) + 1e-8)
    advantages = group_advantages.view(-1) # flatten to align with rolled out groups
    logger.log_group_advantages(group_advantages, counter.global_step)

    right_shifted_rollouts, rollout_attention_mask = old_policy.prepare_rollouts_for_policy_theta(rollouts, rollout_mask)

    batch_overall_loss = 0
    batch_ce_loss = 0 # track this just as a sanity check to make sure model is staying grounded
    # policy update steps
    with autocast(device_type=device, dtype=torch.bfloat16):
        update_epochs = update_config.update_epochs
        for _ in range(update_epochs):
            # generate next token logits at each time step by using rollouts in a teacher forcing step
            theta_logits = policy_theta.decoder(right_shifted_rollouts, img_latent, rollout_attention_mask, latent_attention_mask, checkpoint_grads=True)
            grpo_objective = calc_grpo_objective(theta_logits, rollouts, rollout_attention_mask, old_policy_log_probs, advantages, update_config.epsilon, num_groups)
            entropy_bonus = calc_entropy_bonus(theta_logits, rollout_attention_mask, vocab_size)

            if loss_config.lambda_ce:
                ce_loss = calc_teacher_forced_ce_loss(policy_theta, unexpanded_img_latent, unexpanded_latent_attention_mask, unexpanded_target_lmx_seqs, ce_loss_fn)
            else:
                ce_loss = 0
            logger.log_raw_objective_components(grpo_objective, entropy_bonus, ce_loss, counter.global_step)

            shaped_objective = grpo_objective + loss_config.entropy_beta * entropy_bonus - loss_config.lambda_ce * ce_loss
            loss = -shaped_objective
            batch_overall_loss += loss.item()
            batch_ce_loss += ce_loss.item()
            logger.log_overall_loss(loss, counter.global_step)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_theta.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        counter.global_step += 1

    avg_update_overall_loss = batch_overall_loss / update_epochs
    avg_update_ce_loss = batch_ce_loss / update_epochs
    avg_batch_reward = raw_group_rewards.mean().item()
    avg_batch_reward_components = reward_components.avg_over_rollouts()
    return avg_update_overall_loss, avg_update_ce_loss, avg_batch_reward, avg_batch_reward_components

# average epoch stats over all outer steps. grpo_update() already dealt with differences in whether to average over
# whole batch or multiple update steps within batch
def average_epoch_stats(
    epoch_train_overall_loss, 
    epoch_train_ce_loss, 
    epoch_train_reward, 
    epoch_train_reward_components, 
    num_batches,
    ):

    avg_train_overall_loss = epoch_train_overall_loss / num_batches 
    avg_train_reward = epoch_train_reward / num_batches 
    avg_train_reward_components = epoch_train_reward_components / num_batches
    avg_train_ce_loss = epoch_train_ce_loss / num_batches 

    print(f"Epoch level stats:\nAverage overall train loss over all inner updates: {avg_train_overall_loss}\nAverage train reward over all rollouts: " \
          f"{avg_train_reward}\nAverage Average train reward components over all rollouts: {avg_train_reward_components}\nAverage train teacher forced " \
          f"ce loss over all inner updates: {avg_train_ce_loss}")

    return {
        "avg_train_overall_loss": avg_train_overall_loss,
        "avg_train_reward": avg_train_reward,
        "avg_train_reward_components": avg_train_reward_components,
        "avg_train_ce_loss": avg_train_ce_loss,
    }

def epoch_train_loop(train_dataloader, mini_val_dataloader, old_policy, policy_theta, optimizer, lr_scheduler, grpo_config, curriculum_scheduler, ce_loss_fn, logger, device, counter):
    batch_size = train_dataloader.batch_size
    len_dataset = len(train_dataloader.dataset)
    num_batches = len(train_dataloader)
    mini_val_freq = grpo_config.mini_validation_freq

    # these are epoch level metrics. Within each batch, average over rollouts, steps, exs, etc.,
    # accumulate totals for each loop, then average those at the end for epoch-level averages. We exclude
    # mini validation from here because that's really a diagnostic that doesn't add anything on top of full validation
    epoch_train_overall_loss = 0
    epoch_train_ce_loss = 0
    epoch_train_reward = 0
    epoch_train_reward_components = RewardComponents(0, 0, 0, 0, 0)

    for i, batch in enumerate(train_dataloader):
        logger.log_lr(optimizer, counter.global_step)
        logger.log_configs(grpo_config, counter.global_step)
        # each batch, we need to refresh old_policy to match the resulting updated policy_theta from the previous GRPO updates.
        # Instead of repeatedly deepcopying, and since the encoder and transition head are frozen, we can just refresh the decoder parameters
        old_policy.decoder.load_state_dict(policy_theta.decoder.state_dict())

        avg_update_overall_loss, avg_update_ce_loss, avg_batch_reward, avg_batch_reward_components = grpo_update(old_policy, policy_theta, optimizer, batch, grpo_config, ce_loss_fn, device, logger, counter)
        epoch_train_overall_loss += avg_update_overall_loss
        epoch_train_ce_loss += avg_update_ce_loss
        epoch_train_reward += avg_batch_reward
        epoch_train_reward_components += avg_batch_reward_components

        lr_scheduler.step()
        curriculum_scheduler.step()
        if i % 100 == 0:
            current_ex = i * batch_size + len(batch)
            print(f"[{current_ex:>5d}/{len_dataset:>5d}]")

        if (i + 1) % mini_val_freq == 0:
            print(f"{'-' * 25}\nMini validation")
            policy_theta.eval()
            mini_val_reward, mini_val_reward_components, mini_val_ce_loss = validation_loop(mini_val_dataloader, policy_theta, reward_config, rollout_config, ce_loss_fn, policy_theta.decoder.pad_idx, device)
            print(f"Results:\nAverage mini validation reward: {mini_val_reward}\nAverage mini validation reward " \
                  f"components: {mini_val_reward_components}\nAverage mini validation teacher forced ce loss: {mini_val_ce_loss}\n{'-' * 25}")
            logger.log_mini_validation_stats(mini_val_reward, mini_val_reward_components, mini_val_ce_loss, counter.global_step)
            policy_theta.train() 

        if (i + 1) % grpo_config.checkpoint_freq == 0:
            checkpoint_train_state((CHECKPOINTS_DIR_PATH / f"step_{counter.global_step}_checkpoint.pth"), policy_theta, optimizer, logger)

        counter.global_step += 1

    epoch_stats = average_epoch_stats(epoch_train_overall_loss, epoch_train_ce_loss, epoch_train_reward, epoch_train_reward_components, num_batches)
    return epoch_stats

# can be mini or full validation, depending on which dataloader is passed in
def validation_loop(dataloader, policy_theta, reward_config, rollout_config, ce_loss_fn, pad_idx, device):
    batch_size = dataloader.batch_size
    num_batches = len(dataloader)
    len_dataset = len(dataloader.dataset)
    validation_reward = 0
    validation_reward_components = RewardComponents(0, 0, 0, 0, 0)
    validation_ce_loss = 0

    group_size = 1
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            for i, batch in enumerate(dataloader):
                imgs, target_lmx_seqs, target_musicxml_strs = zip(*batch)
                imgs = [img.to(device) for img in imgs]
                target_lmx_seqs = [target_lmx_seq.to(device) for target_lmx_seq in target_lmx_seqs]
                num_groups = len(imgs)

                img_latent, latent_attention_mask = policy_theta.encoder(imgs)
                img_latent = policy_theta.transition_head(img_latent)
                rollouts, _, rollout_mask = policy_theta.forward_rollout_policy(img_latent, latent_attention_mask, rollout_config.max_actions, rollout_config.top_k, rollout_config.temperature)

                target_lmx_seqs = torch.nested.nested_tensor(target_lmx_seqs, layout=torch.jagged, device=device)
                target_lmx_seqs = target_lmx_seqs.to_padded_tensor(padding=pad_idx)
                raw_rewards, reward_components = reward_rollouts(reward_config, rollouts, rollout_mask, target_lmx_seqs, target_musicxml_strs, num_groups, group_size, policy_theta.decoder.idxs_to_tokens, pad_idx)

                validation_reward += raw_rewards.mean().item()
                validation_reward_components += reward_components.avg_over_rollouts()
                validation_ce_loss += calc_teacher_forced_ce_loss(policy_theta, img_latent, latent_attention_mask, target_lmx_seqs, ce_loss_fn).item()

                if i % 25 == 0:
                    current_ex = i * batch_size + len(batch)
                    print(f"[{current_ex:>5d}/{len_dataset:>5d}]")

    validation_reward = validation_reward / num_batches
    validation_reward_components = validation_reward_components / num_batches
    validation_ce_loss = validation_ce_loss / num_batches
    return validation_reward, validation_reward_components, validation_ce_loss

def checkpoint_train_state(path, policy_theta, optimizer, logger):
    print(f"Saving training state to {path}")
    torch.save({
        "policy_theta": policy_theta.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    
    logger.flush(csv_path=(MODEL_DIR_PATH / "stats.csv"))

if __name__ == "__main__":
    MODEL_DIR_PATH.mkdir()
    CHECKPOINTS_DIR_PATH.mkdir()
    print(f"Created directories {MODEL_DIR_PATH}, {CHECKPOINTS_DIR_PATH}")

    teacher_forced_vitomr, base_img_transform, base_lmx_transform, device = set_up_omr_teacher_force_train()
    encoder = teacher_forced_vitomr.encoder
    transition_head = teacher_forced_vitomr.transition_head
    # remake decoder into variant that supports KV caching so GRPO doesn't take forever
    decoder = teacher_forced_vitomr.decoder.to_cached_version(TRAIN_BATCH_SIZE)

    teacher_forced_state_dict = torch.load(TEACHER_FORCED_STATE_DICT_PATH)

    policy_theta = GRPOViTOMR(encoder, transition_head, decoder, teacher_forced_state_dict)
    policy_theta.to(device)
    print(f"Model architecture\n{'-' * 50}\n{policy_theta}\n")

    print(f"General hyperparameters\n{'-' * 50}\nImage augmentation probability: {AUGMENTATION_P}\n" \
          f"Train batch size: {TRAIN_BATCH_SIZE}\nValidation batch size: {VALIDATION_BATCH_SIZE}\nLearning rate: {LR}\nAdamW betas: {ADAMW_BETAS}, " \
          f"weight decay: {ADAMW_WEIGHT_DECAY}\nEpochs: {EPOCHS}\nMini validation size: {MINI_VALIDATION_SIZE} exs, frequency (in outer steps): " \
          f"{MINI_VALIDATION_FREQ}\nCheckpoint frequency (in outer steps): {CHECKPOINT_FREQ}\nDataloader workers: {NUM_WORKERS}\n")

    # RL is more unstable and teacher force train already included augmentations, so slightly decrease strength
    camera_augment = v2.RandomApply(transforms=[
        v2.GaussianBlur(kernel_size=15, sigma=(0.1, 0.5)),
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

    # make sure to use a list as indices so datasets are indexed with integers and not tensors 
    mini_validation_dataset = Subset(validation_dataset, torch.randint(low=0, high=len(validation_dataset), size=(MINI_VALIDATION_SIZE, )).tolist())

    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=ragged_collate_fn, pin_memory=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=VALIDATION_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=ragged_collate_fn, pin_memory=True)
    mini_validation_dataloader = DataLoader(mini_validation_dataset, batch_size=VALIDATION_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=ragged_collate_fn, pin_memory=True)

    # old_policy should never be changed from eval() or have any of its requires_grad changed
    old_policy = copy.deepcopy(policy_theta).eval()
    for p in old_policy.parameters():
        p.requires_grad = False

    print(f"Scheduler hyperparematers\n{'-' * 50}\nLr warmup steps: {WARMUP_STEPS}\nMinimum lr: {MIN_LR}\nExploration " \
          f"epochs: {EXPLORATION_EPOCHS}\nMax max_actions: {MAX_MAX_ACTIONS}\nMin top_k: {MIN_TOP_K}\nMin softmax " \
          f"temperature: {MIN_TEMPERATURE}\nMax lambda_len: {MAX_LAMBDA_LEN}\nMin entropy beta: {MIN_ENTROPY_BETA}\n" \
          f"Min lambda_ce: {MIN_LAMBDA_CE}\n")

    rollout_config = INITIAL_ROLLOUT_CONFIG
    reward_config = INITIAL_REWARD_CONFIG
    loss_config = INITIAL_LOSS_CONFIG
    update_config = INITIAL_UPDATE_CONFIG
    print(f"Initial GRPO hyperparameters\n{'-' * 50}\nRollout hyperparameters: {rollout_config}\nReward hyperparameters: {reward_config}\nLoss hyperparameters: {loss_config}\nGRPO update hyperparameters: {update_config}\n")
    ce_loss_fn = OMRCELoss(pad_idx=old_policy.decoder.pad_idx, label_smoothing=0.0)

    optimizer = torch.optim.AdamW(policy_theta.parameters(), lr=LR, betas=ADAMW_BETAS, weight_decay=ADAMW_WEIGHT_DECAY)

    epoch_stats_df = pd.DataFrame(columns=["avg_train_overall_loss", "avg_train_reward", "avg_train_reward_components", "avg_train_ce_loss",
                                           "full_val_reward", "full_val_reward_components", "full_val_ce_loss"])

    writer = SummaryWriter(log_dir=LOG_DIR, max_queue=450) # flushes around every 30 minibatches (~15 logs per minibatch)
    logger = GRPOLogger(writer, epoch_stats_df)
    grpo_config = GRPOConfig(rollout_config, reward_config, loss_config, update_config, MINI_VALIDATION_FREQ, CHECKPOINT_FREQ)
    num_steps_per_epoch = len(train_dataloader)
    lr_scheduler = stepwise_cosine_anneal_with_warmup(optimizer, WARMUP_STEPS, EPOCHS, MIN_LR, num_steps_per_epoch)
    curriculum_scheduler = CurriculumScheduler(grpo_config, EXPLORATION_EPOCHS, EPOCHS, num_steps_per_epoch,
                                               MAX_MAX_ACTIONS, MIN_TOP_K, MIN_TEMPERATURE, MAX_LAMBDA_LEN, MIN_ENTROPY_BETA, MIN_LAMBDA_CE)
    counter = StepCounter(0)

    for i in range(EPOCHS):
        print(f"EPOCH {i + 1}\n{'-' * 50}")
        policy_theta.train()
        epoch_start_time = time.perf_counter()
        epoch_stats = epoch_train_loop(train_dataloader, mini_validation_dataloader, old_policy, policy_theta, optimizer, lr_scheduler, grpo_config, curriculum_scheduler, ce_loss_fn, logger, device, counter)
        epoch_end_time = time.perf_counter()
        print(f"Time: {epoch_end_time - epoch_start_time:>0.2f} seconds\n")

        policy_theta.eval()
        print("Validation")
        full_val_reward, full_val_reward_components, full_val_ce_loss = validation_loop(validation_dataloader, policy_theta, reward_config, rollout_config, ce_loss_fn, policy_theta.decoder.pad_idx, device)
        print(f"Results:\nAverage validation reward: {full_val_reward}\nAverage validation reward " \
              f"components: {full_val_reward_components}\nAverage validation teacher forced ce loss: {full_val_ce_loss}\n")

        epoch_stats["full_val_reward"] = full_val_reward
        epoch_stats["full_val_reward_components"] = full_val_reward_components
        epoch_stats["full_val_ce_loss"] = full_val_ce_loss
        logger.log_epoch_level_stats(epoch_stats, counter.global_step)
        logger.update_epoch_stats_df(epoch_stats, i)

    print("Saving final train state")
    checkpoint_train_state(MODEL_DIR_PATH / f"ending_train_state.pth", policy_theta, optimizer, logger)
    model_path = MODEL_DIR_PATH / "vitomr.pth"
    print(f"Saving final model to {model_path}")
    torch.save(policy_theta.state_dict(), model_path)
    logger.flush(csv_path=(MODEL_DIR_PATH / "stats.csv"))
