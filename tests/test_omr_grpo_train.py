import torch
from acai_omr.train.omr_grpo_train import calc_edit_costs, calc_tedn_scores, calc_wellformedness, calc_token_f1
from acai_omr.train.omr_teacher_force_train import set_up_omr_teacher_force_train
from acai_omr.train.datasets import OlimpicDataset
from acai_omr.config import OLIMPIC_SYNTHETIC_ROOT_DIR, LMX_EOS_TOKEN
import pytest

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
    catastrophic = [False, False, False]
    minor = [0, 0, 0]
    scores = calc_wellformedness(catastrophic, minor, device)
    assert torch.all(scores == 1)

    # some minor errors
    catastrophic = [False, False, False]
    minor = [2, 1, 0]
    scores = calc_wellformedness(catastrophic, minor, device)
    assert scores[0] < scores[1] and scores[-1] == 1

    # some catastrophic
    gamma = 3
    catastrophic = [True, False, False]
    minor = [0, 2, 0]
    scores = calc_wellformedness(catastrophic, minor, device, gamma=gamma)
    assert scores[0] == -gamma and scores[1] < 1

    # mixed
    gamma = 3
    catastrophic = [True, False, True]
    minor = [20, 0, 0] # catastrophic errors override everything
    scores = calc_wellformedness(catastrophic, minor, device, gamma=gamma)
    assert torch.equal(scores, torch.tensor([-gamma, 1, -gamma]))

test_args = [
    # basic test with padding to ignore
    (torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]]), torch.tensor([[0, 0, 0, 0], [0, 10, pad_idx, pad_idx]]), (5 / 8), (5 / 6)),
    # test truncation
    (torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]]), torch.tensor([[0, 0, pad_idx], [0, 10, 0]]), (4 / 8), (4 / 5))
]
@pytest.mark.parametrize("rollouts, target_lmx_seqs, expected_precision, expected_recall", test_args)
def test_calc_f1(rollouts, target_lmx_seqs, expected_precision, expected_recall):
    f1 = calc_token_f1(rollouts, target_lmx_seqs, pad_idx)
    assert f1 == (2 * expected_precision * expected_recall / (expected_precision + expected_recall + 1e-8))

if __name__ == "__main__":
    test_calc_tedn_scores()