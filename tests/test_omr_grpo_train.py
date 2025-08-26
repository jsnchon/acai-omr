import torch
from acai_omr.train.omr_grpo_train import calc_edit_costs, calc_tedn_scores, calc_wellformedness, calc_token_f1, calc_n_gram_penalty, calc_repeat_penalty, calc_len_penalty, calc_main_grpo_objective
from acai_omr.train.omr_teacher_force_train import set_up_omr_teacher_force_train
from acai_omr.train.datasets import OlimpicDataset
from acai_omr.config import OLIMPIC_SYNTHETIC_ROOT_DIR
from test_vitomr import VOCAB_LEN
import pytest
import time

tf_vitomr, base_img_transform, base_lmx_transform, device = set_up_omr_teacher_force_train()
pad_idx = tf_vitomr.decoder.pad_idx

def test_calc_tedn_scores():
    dataset = OlimpicDataset(OLIMPIC_SYNTHETIC_ROOT_DIR, "samples.train.txt", base_img_transform, base_lmx_transform, include_musicxml=True)
    img, target_lmx, target_musicxml = dataset[0]
    _, _, other_musicxml = dataset[1]
    idxs_to_tokens = tf_vitomr.decoder.idxs_to_tokens

    batch_size = 2
    num_rollouts_per_img = 3

    # test with well-formed rollouts
    target_musicxml_strs = (target_musicxml, other_musicxml)
    rollouts = torch.stack([target_lmx] * 6)

    edit_costs, catastrophic_errs, minor_errs = calc_edit_costs(rollouts, pad_idx, batch_size, num_rollouts_per_img, target_musicxml_strs, idxs_to_tokens)
    # edit costs for the first 3 rollouts should be 0 since they're the sequences labeled for target_musicxml
    assert all([x == 0 for x in edit_costs[:3]])
    # should have non-zero edit costs for the second 3 since they don't match the used musicxml
    assert all([x != 0 for x in edit_costs[3:]])
    # should have no delinearization errors
    assert sum(catastrophic_errs) == 0 and sum(minor_errs) == 0
    scores = calc_tedn_scores(edit_costs)
    assert all([x == 1 for x in scores[:3]])

    # speed test
    start = time.perf_counter()
    edit_costs, catastrophic_errs, minor_errs = calc_edit_costs(rollouts, pad_idx, batch_size, num_rollouts_per_img, target_musicxml_strs, idxs_to_tokens)
    end = time.perf_counter()
    print(end - start)

    # test some non-catastrophically malformed rollouts
    rollouts[1, 19] = 50
    rollouts[1, 29] = 50
    rollouts[5, 29] = 50
    edit_costs, catastrophic_errs, minor_errs = calc_edit_costs(rollouts, pad_idx, batch_size, num_rollouts_per_img, target_musicxml_strs, idxs_to_tokens)
    assert minor_errs[1] > 0 and minor_errs[-1] > 0 and minor_errs[1] > minor_errs[-1]

    # test some catastrophically malformed rollouts + some non-catastrophically
    rollouts[0, 1] = 88
    rollouts[3, 1] = 88
    edit_costs, catastrophic_errs, minor_errs = calc_edit_costs(rollouts, pad_idx, batch_size, num_rollouts_per_img, target_musicxml_strs, idxs_to_tokens)
    assert catastrophic_errs[0] and catastrophic_errs[3]

    # test with ragged rollouts: shorter one should be further
    rollouts = torch.stack([target_lmx] * 6)
    rollouts[2, 99:] = pad_idx
    edit_costs, catastrophic_errs, minor_errs = calc_edit_costs(rollouts, pad_idx, batch_size, num_rollouts_per_img, target_musicxml_strs, idxs_to_tokens)
    assert torch.argmax(edit_costs) == 2

def test_calc_wellformedness():
    # perfect well-formendness
    catastrophic = torch.tensor([False, False, False])
    minor = torch.tensor([0, 0, 0])
    scores = calc_wellformedness(catastrophic, minor)
    assert torch.all(scores == 1)

    # some minor errors
    catastrophic = torch.tensor([False, False, False])
    minor = torch.tensor([2, 1, 0])
    scores = calc_wellformedness(catastrophic, minor)
    assert scores[0] < scores[1] and scores[-1] == 1

    # some catastrophic
    gamma = 3
    catastrophic = torch.tensor([True, False, False])
    minor = torch.tensor([0, 2, 0])
    scores = calc_wellformedness(catastrophic, minor)
    assert scores[0] == -gamma and scores[1] < 1

    # mixed
    gamma = 3
    catastrophic = torch.tensor([True, False, True])
    minor = torch.tensor([20, 0, 0]) # catastrophic errors override everything
    scores = calc_wellformedness(catastrophic, minor)
    assert torch.equal(scores, torch.tensor([-gamma, 1, -gamma]))

test_args = [
    # basic test with padding to ignore
    (torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]]), torch.tensor([[0, 0, 0, 0], [0, 10, pad_idx, pad_idx]]), 
    torch.tensor([1, (1 / 4)]), torch.tensor([1, (1 / 2)])),
    # test truncation
    (torch.tensor([[20, 0, 0, 0], [0, 0, 0, 0]]), torch.tensor([[0, 0, pad_idx], [0, 10, 0]]), 
    torch.tensor([(1 / 4), (2 / 4)]), torch.tensor([(1 / 2), (2 / 3)]))
]
@pytest.mark.parametrize("rollouts, target_lmx_seqs, expected_precision, expected_recall", test_args)
def test_calc_f1(rollouts, target_lmx_seqs, expected_precision, expected_recall):
    f1 = calc_token_f1(rollouts, target_lmx_seqs, pad_idx)
    assert torch.equal(f1, (2 * expected_precision * expected_recall / (expected_precision + expected_recall + 1e-8)))

test_args = [
    # basic tests with different ns
    (torch.tensor([[0, 0, 0, 0, 0, 5], [5, 6, 7, 8, 5, 5]]), 2, torch.tensor([(1 / 2) * 1, (1 / 2) * 0])),
    (torch.tensor([[0, 0, 0, 0, 0, 5], [5, 6, 7, 8, 5, 5]]), 3, torch.tensor([(1 / 2) * 0, (1 / 2) * 0])),
    # padding to ignore
    (torch.tensor([[0, 0, 0, 0, pad_idx, pad_idx], [5, 5, 0, 5, 5, 5]]), 2, torch.tensor([(1 / 1) * 1, (1 / 2) * 0])),
    (torch.tensor([[0, 0, 0, 5, pad_idx, pad_idx], [5, 5, 0, 5, 5, 5]]), 1, torch.tensor([(1 / 3) * 2, (1 / 5) * 3])),
    # ignore some n-grams containing padding (second ex penalty is 1 since 2 repeat opportunities ignore the 3rd 2-gram)
    (torch.tensor([[0, 0, 0, 0, pad_idx, pad_idx], [5, 5, 5, 5, 0, pad_idx]]), 2, torch.tensor([(1 / 1) * 1, (1 / 1) * 1])),
]
@pytest.mark.parametrize("rollouts, n, expected_penalty", test_args)
def test_n_gram_penalty(rollouts, n, expected_penalty):
    penalty = calc_n_gram_penalty(rollouts, n, pad_idx)
    assert torch.equal(penalty, expected_penalty)

def test_calc_repeat_penalty():
    rollouts = torch.tensor([[5, 0, 5, 5, 0, 0, pad_idx, pad_idx, pad_idx], [5, 6, 5, 6, 3, 7, 3, 4, 5]]) 
    repeat_penalty = calc_repeat_penalty(rollouts, pad_idx)
    assert repeat_penalty[0] > repeat_penalty[1]

def test_len_penalty():
    # perfect match
    rollout_mask = torch.full([2, 100], fill_value=True)
    targets = torch.arange(0, 100).unsqueeze(0).repeat(2, 1)
    penalty = calc_len_penalty(rollout_mask, targets, pad_idx)
    assert torch.equal(penalty, torch.tensor([0, 0]))

    # within threshold
    targets = torch.arange(0, 105).unsqueeze(0).repeat(2, 1)
    penalty = calc_len_penalty(rollout_mask, targets, pad_idx)
    assert torch.equal(penalty, torch.tensor([0, 0]))

    # ignore padding
    rollout_mask = torch.full([2, 100], fill_value=True)
    targets = torch.arange(0, 130).unsqueeze(0).repeat(2, 1)
    targets[:, 105:] = pad_idx
    penalty = calc_len_penalty(rollout_mask, targets, pad_idx)
    assert torch.equal(penalty, torch.tensor([0, 0]))

    rollout_mask = torch.full([2, 130], fill_value=True)
    targets = torch.arange(0, 130).unsqueeze(0).repeat(2, 1)
    targets[0, 105:] = pad_idx
    penalty = calc_len_penalty(rollout_mask, targets, pad_idx)
    assert penalty[0] > 0
    assert penalty[1] == 0

    # clip super large mismatch
    rollout_mask = torch.full([2, 1], fill_value=True)
    targets = torch.arange(0, 130).unsqueeze(0).repeat(2, 1)
    penalty = calc_len_penalty(rollout_mask, targets, pad_idx)
    assert torch.equal(penalty, torch.tensor([1, 1]))

# test grpo calculation with ragged rollouts
def test_calc_grpo():
    num_groups = 2
    total_rollouts = num_groups * 2
    theta_logits = torch.full([total_rollouts, 3, VOCAB_LEN], fill_value=float("-inf"))
    theta_logits[:, :, 0] = 25
    theta_logits[:, :, tf_vitomr.decoder.eos_idx] = 25
    theta_logits[:, :, 5] = 25
    theta_logits[:, :, 200] = 25 # filler just to get probs to an even 1/4
    # softmax will turn these positions into probs of 0.25 each
    rollouts = torch.tensor([[0, 0, 0, tf_vitomr.decoder.eos_idx], [5, 5, 5, tf_vitomr.decoder.eos_idx], [0, 0, tf_vitomr.decoder.eos_idx, pad_idx], [0, tf_vitomr.decoder.eos_idx, pad_idx, pad_idx]])
    rollout_attention_mask = torch.tensor([[False, False, False], [False, False, False], [False, False, True], [False, True, True]])
    # simulate original probability of 0.25 at 0, original probability of 0.1 at 5
    old_policy_log_probs = torch.full_like(rollouts, fill_value=torch.log(torch.tensor([0.25])).item(), dtype=torch.float)
    old_policy_log_probs[1, :] = torch.log(torch.tensor([0.1])).item()
    advantages = torch.full([total_rollouts], fill_value=2)
    advantages[2:] = 1

    # probability ratios should be 0.25 / 0.25 where rollout token is 0, 0.25 / 0.1 where rollout token is 5. Final results will
    # vary depending on length of each rollout
    print(f"Theta logits:\n{theta_logits}\nRollouts:\n{rollouts}\nRollout attn mask:\n{rollout_attention_mask}\nOld policy log probs:\n{old_policy_log_probs}\nAdvantages:\n{advantages}")
    grpo_objective = calc_main_grpo_objective(theta_logits, rollouts, rollout_attention_mask, old_policy_log_probs, advantages, 100, num_groups)

    expected_ratios = torch.full([total_rollouts, 3], fill_value=1.0)
    expected_ratios[1, :] = 0.25 / 0.1 # ratio for 5 tokens
    # 0 out ratios that should've been masked out due to padding
    expected_ratios[2, -1] = 0
    expected_ratios[3, 1:] = 0

    expected_lens = torch.tensor([3, 3, 2, 1]) # expected right shifted lengths
    expected_ratios = advantages.unsqueeze(1) * expected_ratios # multiply by advantage, average over rollout lens
    expected_ratios = expected_ratios.sum(dim=-1) / expected_lens # average over rollouts
    expected_grpo = expected_ratios.sum() / num_groups # average over groups
    assert grpo_objective == expected_grpo
 
if __name__ == "__main__":
    test_calc_grpo()
