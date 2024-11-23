import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import PascalVOC2012Dataset
from loss import YOLOv1Loss
from model import Yolov1
from train import train_fn
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc. 
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16  # 64 in original paper 
WEIGHT_DECAY = 0
EPOCHS = 10
NUM_WORKERS = 4
PIN_MEMORY = True
LOAD_MODEL = False
<<<<<<< HEAD


# set device
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

# Hyperparameters
LR = 2e-5
BATCH_SIZE = 16
EPOCHS = 10
ROOT_DIR = "/Data/VOC2012"
NUM_WORKDERS = 4
LOAD_MODEL_FILE = "overfit.pth.tar"


def main():
    # Initialize dataset and dataloaders
    train_dataset = PascalVOC2012Dataset(root_dir=ROOT_DIR)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize model, loss function, and optimizer
    model = Yolov1(S=7, B=2, C=20).to(DEVICE)
    criterion = YOLOv1Loss(S=7, B=2, C=20).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Load checkpoint if necessary
    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    # Training loop for multiple epochs
    for epoch in range(EPOCHS):
        print(f"Epoch [{epoch+1}/{EPOCHS}]")
        train_fn(train_loader, model, optimizer, criterion, DEVICE)

        # Save model checkpoint
        if epoch % 10 == 0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)

if __name__ == "__main__":
    main()
