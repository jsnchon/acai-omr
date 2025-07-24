from pre_train import pre_train
from models import MAE
from torch.utils.data import Dataset
import torch

class DebugDataset(Dataset):
    def __len__(self):
        return 64

    def __getitem__(self, idx):
        return torch.rand(1, 4, 4), torch.rand(1, 4, 4)

def test_pre_train():
    debug_kwargs = {"num_layers": 1, "num_heads": 1, "mlp_dim": 1}
    debug_mae = MAE(0.75, 2, encoder_hidden_dim=1, decoder_hidden_dim=1, encoder_kwargs=debug_kwargs, decoder_kwargs=debug_kwargs)
    debug_train_dataset = DebugDataset()
    debug_validation_dataset = DebugDataset()
    pre_train(debug_mae, debug_train_dataset, debug_validation_dataset)

if __name__ == "__main__":
    test_pre_train()