import torch
import pytest
from omr_train import EPOCHS, WARMUP_EPOCHS, BASE_LR, MIN_LR, LMX_VOCAB_PATH, PrepareLMXSequence
from utils import cosine_anneal_with_warmup, plot_lr_schedule
from models import OMRDecoder

def test_lr_scheduler():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=BASE_LR)
    scheduler = cosine_anneal_with_warmup(optimizer, WARMUP_EPOCHS, EPOCHS, MIN_LR)
    plot_lr_schedule(scheduler, optimizer, EPOCHS)

test_data = [("measure key:fifths:-7 time", [0, 3, 4, 19, 2]),
             ("tremolo:4 C1", [0, 226, 66, 2])]
@pytest.mark.parametrize("seq, expected", test_data)
def test_prepare_lmx_seq(seq, expected):
    tokens_to_idxs = OMRDecoder(1, LMX_VOCAB_PATH).tokens_to_idxs
    transform = PrepareLMXSequence(tokens_to_idxs)
    assert transform(seq) == expected

if __name__ == "__main__":
    test_lr_scheduler()