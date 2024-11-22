import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import PascalVOC2012Dataset
from loss import YOLOv1Loss
from model import Yolov1

# set device
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

# Hyperparameters
LR = 2e-5
BATCH_SIZE = 16
EPOCHS = 10
ROOT_DIR = "/Data/VOC2012"
NUM_WORKDERS = 4


def main():
    # Initialize dataset and dataloaders
    train_dataset = PascalVOC2012Dataset(root_dir=ROOT_DIR)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKDERS,
        pin_memory=True,
    )

    # Initialize model, loss function, and optimizer
    model = Yolov1(S=7, B=2, C=20).to(device)
    criterion = YOLOv1Loss(S=7, B=2, C=20).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    


if __name__ == "__main__":
    main()
