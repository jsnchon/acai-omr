import torch
from torch.utils.data import DataLoader
from acai_omr.train.datasets import SystemDetectionDataset
from torch.utils.tensorboard import SummaryWriter
from acai_omr.utils.utils import StepCounter, cosine_anneal_with_warmup
from acai_omr.config import SYSTEM_DETECTION_ROOT_DIR
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.amp import autocast
from pathlib import Path
import albumentations as A
import pandas as pd
import time
import contextlib

MODEL_DIR_PATH = Path("system_detector_train")
CHECKPOINTS_DIR_PATH = MODEL_DIR_PATH / "checkpoints"
LOG_DIR = "runs/system_detector"

MIN_WIDTH = 960
MAX_HEIGHT = 1280

TRAINABLE_BACKBONE_LAYERS = 5

# NOTE: to use the pre-trained RPN, we have to have to match the pre-trained model's use of 5 sizes and 3 aspect ratios
# (one size per FPN feature map, 3 aspect ratios per anchor)
anchor_sizes = ((128,), (256,), (512,), (768,), (1024,))
anchor_aspect_ratios = ((0.25, 0.5, 1.0),) * len(anchor_sizes)

ANCHOR_GENERATOR = AnchorGenerator(
    sizes=anchor_sizes,
    aspect_ratios=anchor_aspect_ratios
)

RCNN_KWARGS = {
    "min_size": MIN_WIDTH, 
    "max_size": MAX_HEIGHT,
    "rpn_pre_nms_top_n_train": 500,
    "rpn_post_nms_top_n_train": 300,
    "rpn_pre_nms_top_n_test": 200,
    "rpn_post_nms_top_n_test": 100,
    "box_nms_thresh": 0.5,
}

NUM_CLASSES = 2

AUGMENT_P = 0.5

BASE_LR = 1e-3
FINE_TUNE_BASE_LR = 1e-4
FINE_TUNE_DECAY_FACTOR = 0.8
MIN_LR = 1e-6
SGD_MOMENTUM = 0.9
SGD_WEIGHT_DECAY = 1e-4

BATCH_SIZE = 16
NUM_WORKERS = 26

WARMUP_EPOCHS = 2
EPOCHS = 40
CHECKPOINT_FREQ = 5

def create_param_groups(model, base_lr, fine_tune_base_lr, fine_tune_decay_factor, trainable_backbone_layers):
    param_groups = [
        {"params": model.rpn.parameters(), "lr": base_lr},
        {"params": model.roi_heads.parameters(), "lr": base_lr},
    ]

    backbone_layer_lrs = []

    backbone = model.backbone.body # this is really a ModuleDict

    BACKBONE_LAYERS = [
        (backbone["conv1"], backbone["bn1"]), # consider first conv/batch norm together as the earliest layer
        backbone["layer1"],
        backbone["layer2"],
        backbone["layer3"],
        backbone["layer4"],
    ]
    for i, layer in enumerate(reversed(BACKBONE_LAYERS[-trainable_backbone_layers:])):
        layer_lr = fine_tune_base_lr * (fine_tune_decay_factor) ** i
        backbone_layer_lrs.append(layer_lr)
        
        if isinstance(layer, tuple): # separate components we want to give the same lr
            for module in layer:
                param_groups.append({"params": module.parameters(), "lr": layer_lr})
        else:
            param_groups.append({"params": layer.parameters(), "lr": layer_lr})

    return param_groups, backbone_layer_lrs

def rcnn_collate_fn(batch):
    return list(zip(*batch))

def save_training_state(path, model, scheduler):
    print(f"Saving training state to {path}")
    torch.save({
        "model_state_dict": model.state_dict(),
        "scheduler_state_dict": scheduler.state_dict()
    }, path)

def train_loop(model, dataloader, optimizer, scheduler, device, writer, counter):
    print("Starting training")
    model.train()
    batch_size = dataloader.batch_size
    len_dataset = len(dataloader.dataset)
    num_batches = len(dataloader)
    epoch_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        imgs = [img.to(device, non_blocking=True) for img in batch[0]]
        targets = [{k: torch.tensor(v, dtype=torch.int64).to(device, non_blocking=True) for k, v in target.items()} for target in batch[1]]
        with autocast(device_type=device, dtype=torch.bfloat16):
            loss_dict = model(imgs, targets)
            loss = sum(loss for loss in loss_dict.values())
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        if batch_idx % 10 == 0:
            current_ex = batch_idx * batch_size + len(imgs)
            print(f"[{current_ex:>5d}/{len_dataset:>5d}]")

        writer.add_scalar(f"train/loss", loss.item(), counter.global_step)
        writer.add_scalar(f"train/hyperparams/base_lr", optimizer.param_groups[0]["lr"], counter.global_step)
        writer.add_scalar(f"train/hyperparams/fine_tune_base_lr", optimizer.param_groups[2]["lr"], counter.global_step)
        counter.increment()
 
    avg_loss = epoch_loss / num_batches
    print(f"Average training loss over this epoch: {avg_loss}")
    return avg_loss

def convert_predictions_to_coco(preds, img_ids):
    result = []
    for pred, img_id in zip(preds, img_ids):
        boxes = pred["boxes"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()

        for box, label, score in zip(boxes, labels, scores):
            x_min, y_min, x_max, y_max = box
            w, h = x_max - x_min, y_max - y_min
            coco_box = [float(x_min), float(y_min), float(w), float(h)]

            result.append({
                "image_id": img_id,
                "category_id": int(label),  
                "bbox": coco_box,
                "score": float(score)
            })            

    return result

def evaluation_loop(model, dataloader, device, validation=True):
    model.eval()
    len_dataset = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    if validation:
        split_file = "validation.json"
    else:
        split_file = "test.json"
    with contextlib.redirect_stdout(None):
        ground_truth_coco = COCO(Path(SYSTEM_DETECTION_ROOT_DIR) / split_file)
    # pycocotools demands these fields be set, so give them filler values
    ground_truth_coco.dataset.setdefault("info", {"description": "dummy"})
    ground_truth_coco.dataset.setdefault("licenses", [])

    all_predictions = []
    for batch_idx, (imgs, img_ids) in enumerate(dataloader):
        with torch.no_grad():
            imgs = [img.to(device) for img in imgs]
            with autocast(device_type=device, dtype=torch.bfloat16):
                preds = model(imgs) # list of prediction dictionaries, one for each image

        # convert model outputs to COCO json so we can later run COCO evaluation 
        all_predictions.extend(convert_predictions_to_coco(preds, img_ids))

        if batch_idx % 2 == 0:
            current_ex = batch_idx * batch_size + len(imgs)
            print(f"[{current_ex:>5d}/{len_dataset:>5d}]")

    preds_coco = ground_truth_coco.loadRes(all_predictions)
    coco_eval = COCOeval(ground_truth_coco, preds_coco, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    mAP = coco_eval.stats[0]
    ap50 = coco_eval.stats[1]
    ap75 = coco_eval.stats[2]
    ar10 = coco_eval.stats[7]
 
    print(f"Results\nmAP: {mAP}\nAP@0.5: {ap50}\nAP@0.75: {ap75}\nAR@10: {ar10}")
    return mAP, ap50, ap75, ar10

def set_up_model(device):
    model = fasterrcnn_resnet50_fpn_v2(weights="COCO_V1", trainable_backbone_layers=TRAINABLE_BACKBONE_LAYERS, **RCNN_KWARGS)

    # replace the detection head and anchor generator
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.rpn.anchor_generator = ANCHOR_GENERATOR
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    model.to(device) # important to move the model after making our changes so the replaced parameters are also replaced
    return model

if __name__ == "__main__":
    MODEL_DIR_PATH.mkdir()
    CHECKPOINTS_DIR_PATH.mkdir()
    print(f"Created directories {MODEL_DIR_PATH}, {CHECKPOINTS_DIR_PATH}")
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using device {device}\n")

    model = set_up_model(device)
    print(f"Model architecture\n{'-' * 50}\n{model}")

    augment_transforms = [
        A.Perspective(scale=(0.025, 0.035), p=1), 
        A.GaussianBlur(blur_limit=(15, 20), sigma_limit=(0.8, 2.0), p=1), 
        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), p=1),
        A.GaussNoise(std_range=(0.03, 0.05), p=1)
        ]
    base_transform_list = [
        A.Normalize(mean=0, std=1), 
        A.ToTensorV2()
        ]
    train_transform_list = [A.SomeOf(augment_transforms, n=len(augment_transforms), p=AUGMENT_P)] + base_transform_list
    bbox_params = A.BboxParams(format="pascal_voc", label_fields=["class_labels"])

    train_dataset = SystemDetectionDataset(SYSTEM_DETECTION_ROOT_DIR, "train.json", transform_list=train_transform_list, bbox_params=bbox_params)
    # validation has no augmentation that touches bboxes, so no need to pass bbox_params
    validation_dataset = SystemDetectionDataset(SYSTEM_DETECTION_ROOT_DIR, "validation.json", transform_list=base_transform_list, bbox_params=None, evaluation=True)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=rcnn_collate_fn, pin_memory=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=rcnn_collate_fn, pin_memory=True)

    param_groups, backbone_layer_lrs = create_param_groups(model, BASE_LR, FINE_TUNE_BASE_LR, FINE_TUNE_DECAY_FACTOR, TRAINABLE_BACKBONE_LAYERS)

    optimizer = torch.optim.SGD(param_groups, momentum=SGD_MOMENTUM, weight_decay=SGD_WEIGHT_DECAY)
    scheduler = cosine_anneal_with_warmup(optimizer, WARMUP_EPOCHS, EPOCHS, MIN_LR, len(train_dataloader))

    writer = SummaryWriter(log_dir=LOG_DIR, max_queue=50)
    epoch_stats_df = pd.DataFrame(columns=["train_loss", "mAP", "ap50", "ap75", "ar10" "base_lr", "fine_tune_base_lr"])
    counter = StepCounter()

    print(f"Model architecture\n{'-' * 50}\n{model}")
    params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters count: {params_count}\n") 

    print(f"General hyperparameters\n{'-' * 50}\nEpochs: {EPOCHS}\nWarmup epochs: {WARMUP_EPOCHS}\nCheckpoint frequency: {CHECKPOINT_FREQ} epochs\n" \
          f"Base lr: {BASE_LR}\nFine tune base lr: {FINE_TUNE_BASE_LR}, layer-wise decay factor of {FINE_TUNE_DECAY_FACTOR}\nMinimum lr: {MIN_LR}\n" \
          f"SGD momentum: {SGD_MOMENTUM}, weight decay: {SGD_WEIGHT_DECAY}\nBatch size: {BATCH_SIZE}\n" \
          f"Number of DataLoader workers: {NUM_WORKERS}\nImage augmentation probability: {AUGMENT_P}\n")

    print(f"Backbone fine-tune base lrs by layer: {backbone_layer_lrs}\n")

    print(f"Training for {EPOCHS} epochs. Checkpointing every {CHECKPOINT_FREQ} epochs")
    for i in range(EPOCHS):
        print(f"\nEpoch {i + 1}\n{'-' * 50}")
        base_lr = optimizer.param_groups[0]["lr"] 
        fine_tune_base_lr = optimizer.param_groups[2]["lr"] 
        print(f"Hyperparameters at epoch start:\nBase learning rate: {base_lr:>0.8f}\nFine-tune base learning rate: {fine_tune_base_lr:>0.8f}")

        train_start_time = time.perf_counter()
        epoch_train_loss = train_loop(model, train_dataloader, optimizer, scheduler, device, writer, counter)
        train_end_time = time.perf_counter()
        time_delta = train_end_time - train_start_time
        print(f"Time for this training epoch: {time_delta:>0.2f} seconds ({time_delta / 60:>0.2f} minutes)")

        print("Starting validation")
        mAP, ap50, ap75, ar10 = evaluation_loop(model, validation_dataloader, device, validation=True)

        writer.add_scalar("epoch/train_loss", epoch_train_loss, counter.global_step)
        writer.add_scalars("epoch", {"mAP": mAP, "ap50": ap50, "ap75": ap75, "ar10": ar10}, counter.global_step)
        epoch_stats = [epoch_train_loss, mAP, ap50, ap75, ar10, base_lr, fine_tune_base_lr]
        epoch_stats_df.loc[i] = epoch_stats

        if (i + 1) % CHECKPOINT_FREQ == 0:
            print("Checkpointing model, optimizer, scheduler state dicts")
            checkpoint_path = CHECKPOINTS_DIR_PATH / f"epoch_{i+1}_checkpoint.pth"
            save_training_state(checkpoint_path, model, scheduler)
            print("Saving training stats csv")
            epoch_stats_df.to_csv((MODEL_DIR_PATH / "training_stats.csv"))
            writer.flush()

    print("Saving final training state")
    train_state_path = MODEL_DIR_PATH / "ending_train_state.pth"
    save_training_state(train_state_path, model, scheduler)
    model_path = MODEL_DIR_PATH / "system_detector.pth"
    print(f"Saving final model state dict separately to {model_path}")
    torch.save(model.state_dict(), model_path)
    
    epoch_stats_df.to_csv((MODEL_DIR_PATH / "training_stats.csv"))
    writer.flush()
