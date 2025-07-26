from pre_train import pre_train, PE_MAX_HEIGHT, PE_MAX_WIDTH
from models import MAE, Encoder
from torch.utils.data import Dataset
from utils import show_prediction, PRIMUS_PREPARED_ROOT_DIR, PatchDivisibleResize
from torchvision.transforms import v2
from datasets import PreparedDataset, PreTrainWrapper
import torch

# for debug purposes, hidden dims are also 1
DEBUG_KWARGS = {"num_layers": 1, "num_heads": 1, "mlp_dim": 1}
DEBUG_PATCH_SIZE = 2
DEBUG_MAE = MAE(0.75, DEBUG_PATCH_SIZE, PE_MAX_HEIGHT, PE_MAX_WIDTH, encoder_hidden_dim=1, decoder_hidden_dim=1, encoder_kwargs=DEBUG_KWARGS, decoder_kwargs=DEBUG_KWARGS)

class DebugDataset(Dataset):
    def __len__(self):
        return 64

    def __getitem__(self, idx):
        return torch.rand(1, 4, 4), torch.rand(1, 4, 4)

def test_pre_train():
    debug_train_dataset = DebugDataset()
    debug_validation_dataset = DebugDataset()
    pre_train(DEBUG_MAE, debug_train_dataset, debug_validation_dataset)

def test_encoder_transfer():
    DEBUG_PRETRAINED_MAE_PATH = "debug_pretrained_mae.pth"
    pretrained_mae_state_dict = torch.load(DEBUG_PRETRAINED_MAE_PATH)
    # in MAE instance, encoder part's parameters start with encoder., so filter out those params (and also remove prefix to align names with Encoder class)
    encoder_state_dict = {
        param[len("encoder."):]: value for param, value in pretrained_mae_state_dict.items() if param.startswith("encoder.")
    }
    print(encoder_state_dict)
    # when initializing new Encoder, make sure all the params are the same
    new_encoder = Encoder(patch_size=DEBUG_PATCH_SIZE, hidden_dim=1, **DEBUG_KWARGS)
    new_encoder.load_state_dict(encoder_state_dict)

def test_show_prediction():
    patch_size = 16
    mae = MAE(0.75, 16, PE_MAX_HEIGHT, PE_MAX_WIDTH)

    base_transform = v2.Compose([
        v2.ToImage(), # ToTensor is deprecated
        v2.ToDtype(torch.float32, scale=True),
        PatchDivisibleResize(patch_size),
    ])

    primus = PreparedDataset(PRIMUS_PREPARED_ROOT_DIR, transform=base_transform)
    debug_dataset = PreTrainWrapper(primus)
    show_prediction(mae, debug_dataset[0], patch_size)

if __name__ == "__main__":
    test_pre_train()