import torch
import pytest
from acai_omr.train.omr_teacher_force_train import EPOCHS, WARMUP_EPOCHS, BASE_LR, MIN_LR, LMX_VOCAB_PATH, MAX_LMX_SEQ_LEN, PrepareLMXSequence, omr_teacher_force_train
from acai_omr.train.pre_train import PE_MAX_HEIGHT, PE_MAX_WIDTH
from test_pre_train import DEBUG_PRETRAINED_MAE_PATH, DEBUG_KWARGS, DEBUG_PATCH_SIZE
from acai_omr.utils.utils import cosine_anneal_with_warmup, plot_lr_schedule
from acai_omr.models.models import FineTuneOMREncoder, OMRDecoder, ViTOMR, ScheduledSamplingVITOMR
from torch.utils.data import Dataset

class DebugDataset(Dataset):
    def __len__(self):
        return 64
    
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
    debug_encoder = FineTuneOMREncoder(DEBUG_PATCH_SIZE, PE_MAX_HEIGHT, PE_MAX_WIDTH, 1, hidden_dim=10, **DEBUG_KWARGS)
    debug_decoder = OMRDecoder(MAX_LMX_SEQ_LEN, LMX_VOCAB_PATH, hidden_dim=10, **DEBUG_KWARGS)
    debug_mae_state_dict = torch.load(DEBUG_PRETRAINED_MAE_PATH)
    debug_vitomr = ScheduledSamplingVITOMR(debug_encoder, debug_mae_state_dict, debug_decoder)

    omr_teacher_force_train(debug_vitomr, train_dataset, validation_dataset, "cpu") 

if __name__ == "__main__":
    test_omr_train()
