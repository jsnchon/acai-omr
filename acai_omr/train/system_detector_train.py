import torch
from torch.utils.data import DataLoader
from acai_omr.train.datasets import SystemDetectionDataset
from torch.utils.tensorboard import SummaryWriter
from acai_omr.utils.utils import StepCounter, cosine_anneal_with_warmup
from acai_omr.config import SYSTEM_DETECTION_ROOT_DIR
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torch.amp import autocast
from pathlib import Path
import albumentations as A
import pandas as pd
import time

MODEL_DIR_PATH = Path("system_detector_train")
CHECKPOINTS_DIR_PATH = MODEL_DIR_PATH / "checkpoints"
LOG_DIR = "runs/system_detector"

MIN_WIDTH = 960
MAX_HEIGHT = 1280

TRAINABLE_BACKBONE_LAYERS = 5
ANCHOR_GENERATOR = AnchorGenerator(
    sizes=((128, 256, 512, 768, 1024), ),
    aspect_ratios=((0.1, 0.25, 0.5, 1.0), ),
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

BATCH_SIZE = 4
NUM_WORKERS = 4

GRAD_ACCUMULATION_STEPS = 4

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

def train_loop(model, dataloader, optimizer, scheduler, device, grad_accumulation_steps, writer, counter):
    print("Starting training")
    model.train()
    batch_size = dataloader.batch_size
    len_dataset = len(dataloader.dataset)
    num_batches = len(dataloader)
    epoch_loss = 0
    accumlated_losses = []

    for batch_idx, batch in enumerate(dataloader):
        imgs = [img.to(device) for img in batch[0]]
        targets = [{k: torch.tensor(v).to(device) for k, v in target.items()} for target in batch[1]]
        with autocast(device_type=device, dtype=torch.bfloat16):
            loss_dict = model(imgs, targets)
            loss = sum(loss for loss in loss_dict.values())
        
        epoch_loss += loss.item()
        accumlated_losses.append(loss.item())

        loss.backward()
        
        if batch_idx % 10 == 0:
            current_ex = batch_idx * batch_size + len(imgs)
            print(f"[{current_ex:>5d}/{len_dataset:>5d}]")

        if (batch_idx + 1) % grad_accumulation_steps == 0 or (batch_idx + 1) == num_batches:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            writer.add_scalar(f"train/loss", sum(accumlated_losses) / len(accumlated_losses), counter.global_step)
            writer.add_scalar(f"train/hyperparams/base_lr", optimizer.param_groups[0]["lr"], counter.global_step)
            writer.add_scalar(f"train/hyperparams/fine_tune_base_lr", optimizer.param_groups[2]["lr"], counter.global_step)
            accumlated_losses = []
            counter.increment()

    avg_loss = epoch_loss / num_batches
    print(f"Average training loss over this epoch: {avg_loss}")
    return avg_loss

def validation_loop(model, dataloader, device):
    print("Starting validation")
    model.eval()
    validation_loss = 0
    num_batches = len(dataloader)
    len_dataset = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    for batch_idx, batch in enumerate(dataloader):
        with torch.no_grad():
            imgs = [img.to(device) for img in batch[0]]
            targets = [{k: torch.tensor(v).to(device) for k, v in target.items()} for target in batch[1]]
            with autocast(device_type=device, dtype=torch.bfloat16):
                loss_dict = model(imgs, targets)
                loss = sum(loss for loss in loss_dict.values())
       
        validation_loss += loss.item()
        
        if batch_idx % 2 == 0:
            current_ex = batch_idx * batch_size + len(imgs)
            print(f"[{current_ex:>5d}/{len_dataset:>5d}]")

    avg_loss = validation_loss / num_batches
    print(f"Average validation loss for this epoch: {avg_loss}")
    return avg_loss

if __name__ == "__main__":
    MODEL_DIR_PATH.mkdir()
    CHECKPOINTS_DIR_PATH.mkdir()
    print(f"Created directories {MODEL_DIR_PATH}, {CHECKPOINTS_DIR_PATH}")
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using device {device}\n")

    model = fasterrcnn_resnet50_fpn_v2(weights="COCO_V1", trainable_backbone_layers=TRAINABLE_BACKBONE_LAYERS, **RCNN_KWARGS)
    model.to(device)
    print(f"Model architecture\n{'-' * 50}\n{model}")

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the detection head and anchor generator
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    model.rpn.anchor_generator = ANCHOR_GENERATOR

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
    validation_dataset = SystemDetectionDataset(SYSTEM_DETECTION_ROOT_DIR, "validation.json", transform_list=base_transform_list, bbox_params=None)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=rcnn_collate_fn)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=rcnn_collate_fn)

    param_groups, backbone_layer_lrs = create_param_groups(model, BASE_LR, FINE_TUNE_BASE_LR, FINE_TUNE_DECAY_FACTOR, TRAINABLE_BACKBONE_LAYERS)

    optimizer = torch.optim.SGD(param_groups, momentum=SGD_MOMENTUM, weight_decay=SGD_WEIGHT_DECAY)
    num_effective_batches = -(len(train_dataloader) // -GRAD_ACCUMULATION_STEPS)
    scheduler = cosine_anneal_with_warmup(optimizer, WARMUP_EPOCHS, EPOCHS, MIN_LR, num_effective_batches)

    writer = SummaryWriter(log_dir=LOG_DIR, max_queue=50)
    epoch_stats_df = pd.DataFrame(columns=["train_loss", "validation_loss", "base_lr", "fine_tune_base_lr"])
    counter = StepCounter()

    print(f"Model architecture\n{'-' * 50}\n{model}")
    params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters count: {params_count}\n") 

    print(f"General hyperparameters\n{'-' * 50}\nEpochs: {EPOCHS}\nWarmup epochs: {WARMUP_EPOCHS}\nCheckpoint frequency: {CHECKPOINT_FREQ} epochs\n" \
          f"Base lr: {BASE_LR}\nFine tune base lr: {FINE_TUNE_BASE_LR}, layer-wise decay factor of {FINE_TUNE_DECAY_FACTOR}\nMinimum lr: {MIN_LR}\n" \
          f"SGD momentum: {SGD_MOMENTUM}, weight decay: {SGD_WEIGHT_DECAY}\nBatch size: {BATCH_SIZE}\nGradient accumulation steps: {GRAD_ACCUMULATION_STEPS}\n" \
          f"Number of DataLoader workers: {NUM_WORKERS}\nImage augmentation probability: {AUGMENT_P}\n")

    print(f"Backbone fine-tune base lrs by layer: {backbone_layer_lrs}\n")

    print(f"Training for {EPOCHS} epochs. Checkpointing every {CHECKPOINT_FREQ} epochs")
    for i in range(EPOCHS):
        print(f"\nEpoch {i + 1}\n{'-' * 50}")
        base_lr = optimizer.param_groups[0]["lr"] 
        fine_tune_base_lr = optimizer.param_groups[2]["lr"] 
        print(f"Hyperparameters at epoch start:\nBase learning rate: {base_lr:>0.8f}\nFine-tune base learning rate: {fine_tune_base_lr:>0.8f}")

        train_start_time = time.perf_counter()
        epoch_train_loss = train_loop(model, train_dataloader, optimizer, scheduler, device, GRAD_ACCUMULATION_STEPS, writer, counter)
        train_end_time = time.perf_counter()
        time_delta = train_end_time - train_start_time
        print(f"Time for this training epoch: {time_delta:>0.2f} seconds ({time_delta / 60:>0.2f} minutes)")

        epoch_validation_loss = validation_loop(model, validation_dataloader, device)

        writer.add_scalars("epoch", {"train_loss": epoch_train_loss, "validation_loss": epoch_validation_loss}, counter.global_step)
        epoch_stats = [epoch_train_loss, epoch_validation_loss, base_lr, fine_tune_base_lr]
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
