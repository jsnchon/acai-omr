import torch
from acai_omr.config import GRAND_STAFF_ROOT_DIR, OLIMPIC_SYNTHETIC_ROOT_DIR, OLIMPIC_SCANNED_ROOT_DIR
from acai_omr.utils.utils import show_mae_prediction, show_vitomr_prediction, ragged_collate_fn
from pathlib import Path
from models import MAELoss, OMRLoss
from acai_omr.train.datasets import GrandStaffLMXDataset, OlimpicDataset, OlimpicPreTrainWrapper, GrandStaffPreTrainWrapper, GrandStaffOMRTrainWrapper
from torch.utils.data import ConcatDataset, DataLoader
from acai_omr.train.pre_train import set_up_mae, base_transform
from acai_omr.train.omr_train import set_up_omr_train
from torch.amp import autocast
from enum import Enum
import argparse

class Models(Enum):
    MAE = "mae"
    VIT_OMR = "vitomr"

def test_loop(model, model_type, dataloader, loss_fn, device):
    print("Starting evaluation")
    model.eval()
    test_loss = 0

    num_batches = len(dataloader)
    len_dataset = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    with torch.no_grad():
        with autocast(device_type=device, dtype=torch.bfloat16):
            for batch_idx, batch in enumerate(dataloader):
                batch = [(x.to(device, non_blocking=True), y.to(device, non_blocking=True)) for x, y in batch]
                if model_type == Models.MAE:
                    pred, loss_mask, target = model(batch)
                    test_loss += loss_fn(pred, loss_mask, target).item()
                elif model_type == Models.VIT_OMR:
                    pred, target_seqs = model(batch)
                    test_loss += loss_fn(pred, target_seqs).item()

                if batch_idx % 25 == 0:
                    current_ex = batch_idx * batch_size + len(batch)
                    print(f"[{current_ex:>5d}/{len_dataset:>5d}]")
            
    avg_loss = test_loss / num_batches
    print(f"Average test loss: {avg_loss}")

def sample_predictions(model, model_type, prediction_dir, test_dataset, device):
    num_predictions = args.num_predictions
    assert num_predictions is not None, "If a prediction directory is specified, a number of predictions to sample must be specified"

    print(f"Creating directory at {prediction_dir} if it already doesn't exist")
    prediction_dir = Path(prediction_dir)
    prediction_dir.mkdir(exist_ok=True)

    samples = torch.randint(0, len(test_dataset), (num_predictions, ))
    for sample_num, sample in enumerate(samples):
        ex = test_dataset[sample.item()]
        ex = (ex[0].to(device), ex[1].to(device))
        if model_type == Models.MAE:
            save_path = prediction_dir / f"sample_{sample_num}.png"
            show_mae_prediction(model, ex, mae.patch_size, save_path)
        elif model_type == Models.VIT_OMR:
            save_path = prediction_dir / f"sample_{sample_num}"
            show_vitomr_prediction(model, ex, save_path)

def test_mae(mae, mae_state_dict, args, device):
    print("Creating MAE model from loaded state dict")
    mae.load_state_dict(mae_state_dict)
    print("Model architecture\n--------------------")
    print(mae)

    print("Setting up test dataset and dataloader")
    grand_staff = GrandStaffLMXDataset(GRAND_STAFF_ROOT_DIR, "samples.test.txt", img_transform=base_transform)
    olimpic_synthetic = OlimpicDataset(OLIMPIC_SYNTHETIC_ROOT_DIR, "samples.test.txt", img_transform=base_transform)
    olimpic_scanned = OlimpicDataset(OLIMPIC_SCANNED_ROOT_DIR, "samples.test.txt", img_transform=base_transform)

    test_dataset = ConcatDataset([
        GrandStaffPreTrainWrapper(grand_staff),
        OlimpicPreTrainWrapper(olimpic_synthetic),
        OlimpicPreTrainWrapper(olimpic_scanned)
    ])

    batch_size = args.batch_size
    num_workers = args.num_workers
    print(f"Using a batch size of {batch_size} and {num_workers} workers")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=ragged_collate_fn, pin_memory=True)
    loss_fn = MAELoss()

    mae = mae.to(device)

    test_loop(mae, Models.MAE, test_dataloader, loss_fn, device)

    prediction_dir = args.prediction_dir
    if prediction_dir:
        sample_predictions(mae, Models.MAE, prediction_dir, test_dataloader.dataset, device)

def test_vitomr(vitomr, vitomr_state_dict, args, device):
    print("Creating ViTOMR model from loaded state dict")
    vitomr.load_state_dict(vitomr_state_dict)
    print("Model architecture\n--------------------")
    print(vitomr)

    print("Setting up test dataset and dataloader")
    grand_staff = GrandStaffLMXDataset(GRAND_STAFF_ROOT_DIR, "samples.test.txt", img_transform=base_img_transform, lmx_transform=base_lmx_transform)
    olimpic_synthetic = OlimpicDataset(OLIMPIC_SYNTHETIC_ROOT_DIR, "samples.test.txt", img_transform=base_img_transform, lmx_transform=base_lmx_transform)
    olimpic_scanned = OlimpicDataset(OLIMPIC_SCANNED_ROOT_DIR, "samples.test.txt", img_transform=base_img_transform, lmx_transform=base_lmx_transform)

    test_dataset = ConcatDataset([
        GrandStaffOMRTrainWrapper(grand_staff),
        olimpic_synthetic,
        olimpic_scanned,
    ])

    batch_size = args.batch_size
    num_workers = args.num_workers
    print(f"Using a batch size of {batch_size} and {num_workers} workers")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=ragged_collate_fn, pin_memory=True)
    loss_fn = OMRLoss(vitomr.decoder.padding_idx, label_smoothing=0.0)

    vitomr = vitomr.to(device)

    test_loop(vitomr, Models.VIT_OMR, test_dataloader, loss_fn, device)

    prediction_dir = args.prediction_dir
    if prediction_dir:
        sample_predictions(vitomr, Models.VIT_OMR, prediction_dir, test_dataloader.dataset, device)

mae = set_up_mae()
vitomr, base_img_transform, base_lmx_transform, _ = set_up_omr_train()

parser = argparse.ArgumentParser()
parser.add_argument("model_type", choices=[Models.MAE.value, Models.VIT_OMR.value])
parser.add_argument("weight_path", help="Path to .pth weight or checkpoint file")
parser.add_argument("-c", "--checkpoint", action="store_true", help="Using a checkpoint (not just standalone model weight file)")
parser.add_argument("-d", "--prediction-dir", type=str, default=None, 
                    help="If specified, sample some examples and save the model's predictions to this directory")
parser.add_argument("-p", "--num-predictions", type=int, default=None,
                    help="Number of predictions to sample and save (required if prediction-dir is specified)")
parser.add_argument("-b", "--batch-size", type=int, default=64)
parser.add_argument("-w", "--num-workers", type=int, default=24)

args = parser.parse_args()
model_type = args.model_type

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using device {device}")

weight_path = args.weight_path
print(f"Loading state dict from {weight_path}")
if args.checkpoint:
    if device == "cpu":
        checkpoint = torch.load(weight_path, map_location=torch.device("cpu"))
    else:
        checkpoint = torch.load(weight_path)
    if args.model_type == Models.MAE:
        model_state_dict = checkpoint["mae_state_dict"]
    elif args.model_type == Models.VIT_OMR:
        model_state_dict = checkpoint["vitomr_state_dict"]
else:
    if device == "cpu":
        model_state_dict = torch.load(weight_path, map_location=torch.device("cpu"))
    else:
        model_state_dict = torch.load(weight_path)

if model_type == Models.MAE.value: # use Enum.value here since those strings (not actual Enum objects) are the command line options
    test_mae(mae, model_state_dict, args, device)
elif model_type == Models.VIT_OMR.value:
    test_vitomr(vitomr, model_state_dict, args, device)
