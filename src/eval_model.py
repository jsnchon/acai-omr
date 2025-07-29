import torch
from utils import show_prediction, GRAND_STAFF_ROOT_DIR, OLIMPIC_SYNTHETIC_ROOT_DIR, OLIMPIC_SCANNED_ROOT_DIR
from pathlib import Path
from models import MAELoss
from datasets import GrandStaffLMXDataset, OlimpicDataset, OlimpicPreTrainWrapper, GrandStaffPreTrainWrapper
from torch.utils.data import ConcatDataset, DataLoader
from pre_train import mae, base_transform, pre_train_collate_fn
import argparse

def test_loop(mae, dataloader, loss_fn, device):
    print("Starting evaluation")
    mae.eval()
    test_loss = 0

    num_batches = len(dataloader)
    len_dataset = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = [(x.to(device, non_blocking=True), y.to(device, non_blocking=True)) for x, y in batch]
            pred, loss_mask, target = mae(batch)
            test_loss += loss_fn(pred, loss_mask, target).item()

            if batch_idx % 25 == 0:
                current_ex = batch_idx * batch_size + len(batch)
                print(f"[{current_ex:>5d}/{len_dataset:>5d}]")
        
    avg_loss = test_loss / num_batches
    print(f"Average test loss: {avg_loss}")
    return avg_loss

parser = argparse.ArgumentParser()
parser.add_argument("weight_path", help="Path to .pth weight or checkpoint file")
parser.add_argument("-c", "--checkpoint", action="store_true", help="Using a checkpoint (not just standalone model weight file)")
parser.add_argument("-d", "--prediction-dir", type=str, default=None, 
                    help="If specified, sample some examples and save the model's predictions to this directory")
parser.add_argument("-p", "--num-predictions", type=int, default=None,
                    help="Number of predictions to sample and save (required if prediction-dir is specified)")
parser.add_argument("-b", "--batch-size", type=int, default=64)
parser.add_argument("-w", "--num-workers", type=int, default=24)

args = parser.parse_args()

weight_path = args.weight_path
print(f"Loading state dict from {weight_path}")
if args.checkpoint:
    checkpoint = torch.load(weight_path)
    mae_state_dict = checkpoint["mae_state_dict"]
else:
    mae_state_dict = torch.load(weight_path)

print("Creating MAE model from loaded state dict")
mae.load_state_dict(mae_state_dict)
print("Model architecture\n--------------------")
print(mae)

print("Setting up test dataset and dataloader")
grand_staff = GrandStaffLMXDataset(GRAND_STAFF_ROOT_DIR, "samples.test.txt", transform=base_transform)
olimpic_synthetic = OlimpicDataset(OLIMPIC_SYNTHETIC_ROOT_DIR, "samples.test.txt", transform=base_transform)
olimpic_scanned = OlimpicDataset(OLIMPIC_SCANNED_ROOT_DIR, "samples.test.txt", transform=base_transform)

test_dataset = ConcatDataset([
    GrandStaffPreTrainWrapper(grand_staff),
    OlimpicPreTrainWrapper(olimpic_synthetic),
    OlimpicPreTrainWrapper(olimpic_scanned)
])

batch_size = args.batch_size
num_workers = args.num_workers
print(f"Using a batch size of {batch_size} and {num_workers} workers")
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=pre_train_collate_fn, pin_memory=True)
loss_fn = MAELoss()

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using device {device}")

mae = mae.to(device)

test_loop(mae, test_dataloader, loss_fn, device)

prediction_dir = args.prediction_dir
if prediction_dir:
    num_predictions = args.num_predictions
    assert num_predictions is not None, "If a prediction directory is specified, a number of predictions to sample must be specified"

    print(f"Creating directory at {prediction_dir} if it already doesn't exist")
    prediction_dir = Path(prediction_dir)
    prediction_dir.mkdir(exist_ok=True)

    samples = torch.randint(0, len(test_dataset), (num_predictions, ))
    for sample_num, sample in enumerate(samples):
        save_path = prediction_dir / f"sample_{sample_num}.png"
        ex = test_dataset[sample.item()]
        ex = (ex[0].to(device), ex[1].to(device))
        show_prediction(mae, ex, mae.patch_size, save_path)