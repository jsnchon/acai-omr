import torch
from torch.amp import autocast
from acai_omr.train.system_detector_train import set_up_model
from pathlib import Path
from torchvision.transforms import v2
from PIL import Image
import json

MODEL_PATH = ""
IMAGE_DIR = Path("") # should only contain images
CONFIDENCE_THRESHOLD = 0.5
MERGED_JSON_PATH = ""

print(f"Running inference on directory {IMAGE_DIR} using model saved at {MODEL_PATH}. Adding annotations with a confidence threshold of {CONFIDENCE_THRESHOLD}")

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using device {device}")

model = set_up_model(device)
model.to(device)
print(model)

print(f"Loading state dict from {MODEL_PATH}")
model_state_dict = torch.load(MODEL_PATH)
model.load_state_dict(model_state_dict)

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

# load json file containing the entire unsplit dataset
with open(MODEL_PATH / "system_detection.json", "r") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]

max_img_id = max(img["id"] for img in images) 
max_ann_id = max(ann["id"] for ann in annotations) 

model.eval()
new_anns_count = 0
for i, img_file in enumerate(IMAGE_DIR.iterdir()):
    img = Image.open(img_file).convert("L")
    img = transform(img).to(device, non_blocking=True)

    h = img.shape[1]
    w = img.shape[2]
    img_id = max_img_id + i

    images.append({
        "id": img_id,
        "file_name": str(img_file),
        "width": w,
        "height": h
    })

    with torch.no_grad():
        with autocast(device_type=device, dtype=torch.bfloat16):
            pred = model([img])[0]

    boxes = pred["boxes"].cpu().numpy()
    labels = pred["labels"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()

    for box, label, score in zip(boxes, labels, scores):
        if score < CONFIDENCE_THRESHOLD:
            continue
        x_min, y_min, x_max, y_max = box
        coco_box = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)] 
        ann_id = max_ann_id + 1 + new_anns_count

        annotations.append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": int(label),  
            "bbox": coco_box,
            "iscrowd": 0
        })

        new_anns_count += 1

coco["images"] = images
coco["annotations"] = annotations

with open(MERGED_JSON_PATH, "w") as f:
    json.dump(coco, f)

print(f"Saved new COCO dataset json to {MERGED_JSON_PATH}")
