import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from acai_omr.config import SYSTEM_DETECTION_ROOT_DIR
from acai_omr.train.datasets import SystemDetectionDataset
from acai_omr.train.system_detector_train import set_up_model, rcnn_collate_fn
import albumentations as A

MODEL_PATH = ""
BATCH_SIZE = 8
NUM_WORKERS = 12

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using device {device}")

model = set_up_model(device)
print(model)

print(f"Loading state dict from {MODEL_PATH}")
model_state_dict = torch.load(MODEL_PATH)
model.load_state_dict(model_state_dict)

print(f"Setting up test dataset and dataloader with a batch size of {BATCH_SIZE} and {NUM_WORKERS} workers")
transform_list = [
    A.Normalize(mean=0, std=1), 
    A.ToTensorV2()
    ]
bbox_params = A.BboxParams(format="pascal_voc", label_fields=["class_labels"])
test_dataset = SystemDetectionDataset(SYSTEM_DETECTION_ROOT_DIR, "test.json", transform_list=transform_list, bbox_params=bbox_params)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=rcnn_collate_fn, pin_memory=True)

print("Starting evaluation")
model.eval()
test_loss = 0

num_batches = len(test_dataloader)
len_dataset = len(test_dataloader.dataset)
batch_size = test_dataloader.batch_size

for batch_idx, batch in enumerate(test_dataloader):
    imgs = [img.to(device, non_blocking=True) for img in batch[0]]
    targets = [{k: torch.tensor(v).to(device, non_blocking=True) for k, v in target.items()} for target in batch[1]]
    with torch.no_grad():
        with autocast(device_type=device, dtype=torch.bfloat16):
            loss_dict = model(imgs, targets)
            loss = sum(loss for loss in loss_dict.values())
        
    test_loss += loss.item()

    if batch_idx % 2 == 0:
        current_ex = batch_idx * batch_size + len(imgs)
        print(f"[{current_ex:>5d}/{len_dataset:>5d}]")

    avg_loss = test_loss / num_batches
    print(f"Average test loss: {avg_loss}")
