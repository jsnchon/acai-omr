import torch
import pytest
from omr_train import EPOCHS, WARMUP_EPOCHS, BASE_LR, MIN_LR, LMX_VOCAB_PATH, MAX_LMX_SEQ_LEN, PrepareLMXSequence, omr_train
from pre_train import PE_MAX_HEIGHT, PE_MAX_WIDTH
from test_pre_train import DEBUG_PRETRAINED_MAE_PATH, DEBUG_KWARGS, DEBUG_PATCH_SIZE
from utils import cosine_anneal_with_warmup, plot_lr_schedule
from models import FineTuneOMREncoder, OMRDecoder, ViTOMR
from torch.utils.data import Dataset

class DebugDataset(Dataset):
    def __len__(self):
        return 256
    
    def __getitem__(self, idx):
        return torch.rand(1, 32, 32), torch.randint(10, 100, (10, ))

def test_lr_scheduler():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=BASE_LR)
    scheduler = cosine_anneal_with_warmup(optimizer, WARMUP_EPOCHS, EPOCHS, MIN_LR)
    plot_lr_schedule(scheduler, optimizer, EPOCHS)

test_data = [("measure key:fifths:-7 time", torch.tensor([0, 3, 4, 19, 2])),
             ("tremolo:4 C1", torch.tensor([0, 226, 66, 2]))]
@pytest.mark.parametrize("seq, expected", test_data)
def test_prepare_lmx_seq(seq, expected):
    tokens_to_idxs = OMRDecoder(1, LMX_VOCAB_PATH).tokens_to_idxs
    transform = PrepareLMXSequence(tokens_to_idxs)
    assert torch.equal(transform(seq), expected)

def test_omr_train():
    train_dataset = DebugDataset()
    validation_dataset = DebugDataset()
    debug_encoder = FineTuneOMREncoder(DEBUG_PATCH_SIZE, PE_MAX_HEIGHT, PE_MAX_WIDTH, 2, hidden_dim=10, **DEBUG_KWARGS)
    debug_decoder = OMRDecoder(MAX_LMX_SEQ_LEN, LMX_VOCAB_PATH, hidden_dim=10, **DEBUG_KWARGS)
    debug_mae_state_dict = torch.load(DEBUG_PRETRAINED_MAE_PATH)
    debug_vitomr = ViTOMR(debug_encoder, debug_mae_state_dict, debug_decoder)

    omr_train(debug_vitomr, train_dataset, validation_dataset, "cpu") 

if __name__ == "__main__":
    test_omr_train()