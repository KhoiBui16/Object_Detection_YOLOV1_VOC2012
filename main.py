import os
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import PascalVOC2012Dataset
from loss import YOLOv1Loss
from model import Yolov1
from train import train_fn
from dataset import custom_collate_fn
from train import Compose
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
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 20  # 64 in original paper 
WEIGHT_DECAY = 0
EPOCHS = 1
NUM_WORKERS = 7
PIN_MEMORY = True
LOAD_MODEL = True
LOAD_MODEL_FILE = "overfit.pth.tar"
ROOT_DIR = "\\Data\\VOC2012"
IMG_SIZE = 448

# set device
device = torch.device("cuda" if torch.cuda.is_available else "cpu")


def main():
    # Initialize dataset and dataloaders
    train_dataset = PascalVOC2012Dataset(
        root_dir=ROOT_DIR,
        img_size=IMG_SIZE,
        transforms=Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),]),
        S=7,
        B=2,
        C=20,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )

    # Initialize model, loss function, and optimizer
    model = Yolov1(S=7, B=2, C=20).to(DEVICE)
    criterion = YOLOv1Loss(S=7, B=2, C=20).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Check if the checkpoint file exists
    if LOAD_MODEL:
        if os.path.exists(LOAD_MODEL_FILE):
            print(f"Loading checkpoint from {LOAD_MODEL_FILE}...")
            load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)
        else:
            print(f"Checkpoint file '{LOAD_MODEL_FILE}' not found. Creating initial checkpoint...")
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": 0,  # Bắt đầu từ epoch 0
            }
            torch.save(checkpoint, LOAD_MODEL_FILE)
            print(f"Initial checkpoint saved to {LOAD_MODEL_FILE}.")

    # Training loop for multiple epochs
    for epoch in range(EPOCHS):
        print(f"Epoch [{epoch+1}/{EPOCHS}]")
        
        # Train model and get mean loss for this epoch
        train_loss = train_fn(train_loader, model, optimizer, criterion, DEVICE)

        # Save model checkpoint
        if epoch % 10 == 0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=f"model_epoch_{epoch+1}.pt")
        
        # Print training loss for this epoch
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.4f}")

        # Đánh giá sau mỗi epoch
        all_pred_boxes, all_true_boxes = get_bboxes(train_loader, model, iou_threshold=0.5, threshold=0.4)
        mAP = mean_average_precision(all_pred_boxes, all_true_boxes, iou_threshold=0.5, box_format="midpoint")
        print(f"Mean Average Precision (mAP) at epoch {epoch + 1}: {mAP}")

        # Plot one image after each epoch with its predicted bounding boxes
        if epoch % 1 == 0:  # Plot every epoch, or change to another condition
            # Select an image from the batch and the corresponding predicted boxes
            x, labels = next(iter(train_loader))  # Get one batch from the loader
            x = x.to(DEVICE)
            labels = labels['yolo_targets'].to(DEVICE)

            # Make prediction with the model
            with torch.no_grad():
                predictions = model(x)

            # Convert predictions to bounding boxes (detected boxes after NMS)
            bboxes = cellboxes_to_boxes(predictions)
            nms_boxes = non_max_suppression(
                bboxes[0], iou_threshold=0.5, threshold=0.4, box_format="midpoint"
            )
            
            # Giả sử x[0] có dạng tensor [C, H, W]
            img = x[0].permute(1, 2, 0).cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min())  # Chuẩn hóa về [0, 1]
            plot_image(img, nms_boxes)


if __name__ == "__main__":
    main()

