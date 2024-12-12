import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import PascalVOC2012Dataset, collate_fn
from model import Yolov1
from loss import YOLOv1Loss
from train import train_model, validate_model, load_checkpoint, save_checkpoint

seed = 123
torch.manual_seed(seed)

# Hyperparameter
C = 20
B = 2
S = 7
LEARNING_RATE  = 1e-4
BATCH_SIZE     = 4
EPOCHS         = 1
NUM_WORKERS    = 2
PIN_MEMORY     = True
IMG_SIZE       = 448
WEIGHT_DECAY   = 5e-4
MOMENTUM       = 0.9
LOAD_MODEL     = True
ROOT_DIR       = r"Data/VOC2012" 
CHECKPOINT_PATH = r"./best_model.pth"
CHECKPOINT_FILE = "best_model.pth"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    # initialize train_dataset
    train_dataset = PascalVOC2012Dataset(
        root_dir = ROOT_DIR,
        split    = "train",
        S        = S,
        B        = B,
        C        = C
    )

    # initialize train_loader
    train_loader = DataLoader(
        train_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        num_workers = NUM_WORKERS,
        pin_memory  = PIN_MEMORY,
        collate_fn  = collate_fn,
    )

    # initialize eval_dataset
    valid_dataset = PascalVOC2012Dataset(
        root_dir = ROOT_DIR,
        split    = "val",
        S        = S,
        B        = B,
        C        = C
    )

    # initialize train_loader
    val_loader = DataLoader(
        valid_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = NUM_WORKERS,
        pin_memory  = PIN_MEMORY,
        collate_fn  = collate_fn,
    )

    # initialize model, loss, opimizer
    model     = Yolov1(split_size=S, num_boxes=B, num_classes=C).to(DEVICE)
    criterion = YOLOv1Loss(S=S,B=B, C=C).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    # Load model from checkpoint if LOAD_MODEL is True
    start_epoch = 0
    if LOAD_MODEL:
        if os.path.exists(CHECKPOINT_PATH):
            print(f"Checkpoint {CHECKPOINT_FILE} found. \nLoading {CHECKPOINT_FILE}...")
            model, optimizer, start_epoch, _ = load_checkpoint(CHECKPOINT_PATH, model, optimizer)
        else:
            print(f"Checkpoint {CHECKPOINT_PATH} not found. Creating new checkpoint.")
            save_checkpoint(model, optimizer, start_epoch, loss=0.0, filepath=CHECKPOINT_PATH)

    # Training and validation loop
    best_loss = float("inf")
    for epoch in range(start_epoch, EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        train_loss = train_model(model, train_loader, criterion, optimizer, DEVICE, epoch)
        val_loss = validate_model(model, val_loader, criterion, DEVICE)
        print(f"Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

        # Save checkpoint if validation loss improves
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, filepath="best_model.pth")

if __name__ == '__main__':
    main()