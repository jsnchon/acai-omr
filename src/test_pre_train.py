from pre_train import pre_train, PE_MAX_HEIGHT, PE_MAX_WIDTH, base_transform
from models import MAE, Encoder
from torch.utils.data import Dataset
from utils import show_prediction, PRIMUS_PREPARED_ROOT_DIR
from torchvision.transforms import v2, InterpolationMode
from datasets import PreparedDataset, PreTrainWrapper
import torch
from pre_train import mae as pre_train_mae

# for debug purposes, hidden dims are also 1
DEBUG_KWARGS = {"num_layers": 1, "num_heads": 1, "mlp_dim": 1}
DEBUG_PATCH_SIZE = 16
DEBUG_MAE = MAE(0.75, DEBUG_PATCH_SIZE, PE_MAX_HEIGHT, PE_MAX_WIDTH, encoder_hidden_dim=1, decoder_hidden_dim=1, encoder_kwargs=DEBUG_KWARGS, decoder_kwargs=DEBUG_KWARGS)

class DebugDataset(Dataset):
    def __len__(self):
        return 64

    def __getitem__(self, idx):
        return torch.rand(1, 32, 32), torch.rand(1, 32, 32)

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

    primus = PreparedDataset(PRIMUS_PREPARED_ROOT_DIR, transform=base_transform)
    debug_dataset = PreTrainWrapper(primus)
    show_prediction(mae, debug_dataset[0], patch_size)

# qualitatively evaluate the model is learning to do what it needs to. scp a checkpoint file then pass it in here
# assumes will run the image on cpu
def basic_prediction_test(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    mae_state_dict = checkpoint["mae_state_dict"]
    pre_train_mae.load_state_dict(mae_state_dict)

    primus = PreparedDataset(PRIMUS_PREPARED_ROOT_DIR, transform=base_transform)
    debug_dataset = PreTrainWrapper(primus)
    sample = torch.randint(0, len(debug_dataset), (1, )).item()
    show_prediction(pre_train_mae, debug_dataset[sample], 16, "unaugmented_prediction.png")

    camera_augment = v2.Compose([
        v2.GaussianBlur(kernel_size=15, sigma=1),
        v2.GaussianNoise(sigma=0.03),
        v2.RandomRotation(degrees=(-1, 1), interpolation=InterpolationMode.BILINEAR),
        v2.RandomPerspective(distortion_scale=0.06, p=1),
        v2.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.2, hue=0),
    ])

    augmented_debug_dataset = PreTrainWrapper(primus, transform=camera_augment)
    show_prediction(pre_train_mae, augmented_debug_dataset[sample], 16, "augmented_prediction.png")

if __name__ == "__main__":
    # test_pre_train()
    basic_prediction_test("epoch_150_checkpoint.pth")