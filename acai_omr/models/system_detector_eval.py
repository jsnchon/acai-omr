import torch
from torch.utils.data import DataLoader
from acai_omr.config import SYSTEM_DETECTION_ROOT_DIR
from acai_omr.train.datasets import SystemDetectionDataset
from acai_omr.train.system_detector_train import set_up_model, rcnn_collate_fn, evaluation_loop
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
evaluation_loop(model, test_dataloader, device, validation=False)
