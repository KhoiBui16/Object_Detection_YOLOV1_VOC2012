import os
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import PascalVOC2012Dataset, collate_fn
from model import Yolov1
from loss import YOLOv1Loss
from utils import train_model, validate_model, load_checkpoint, save_checkpoint

seed = 123
torch.manual_seed(seed)

# Hyperparameters
C = 20
B = 2
S = 7
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
EPOCHS = 10
NUM_WORKERS = 2
PIN_MEMORY = True
IMG_SIZE = 448
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
LOAD_MODEL = True
ROOT_DIR = "Data/VOC2012"
CHECKPOINT_DIR = "Checkpoints"
BEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def main():

    # initialize train_dataset
    train_dataset = PascalVOC2012Dataset(
        root_dir=ROOT_DIR, split="train", S=S, B=B, C=C, transform=transform
    )

    # initialize train_loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_fn,
    )

    # initialize eval_dataset
    valid_dataset = PascalVOC2012Dataset(
        root_dir=ROOT_DIR, split="val", S=S, B=B, C=C, transform=transform
    )

    # initialize train_loader
    val_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_fn,
    )

    # initialize model, loss, opimizer
    model = Yolov1(split_size=S, num_boxes=B, num_classes=C).to(DEVICE)
    criterion = YOLOv1Loss(S=S, B=B, C=C).to(DEVICE)
    
    """ optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    ) """
    
    optimizer = optim.Adam(
        model.parameters(),
        lr = LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Load model from checkpoint if LOAD_MODEL is True
    start_epoch = 0

    if LOAD_MODEL:
        # Tạo thư mục checkpoint nếu chưa tồn tại
        if not os.path.exists(CHECKPOINT_DIR):
            print(f"Not found checkpoint_dir at [{CHECKPOINT_DIR}] !!!")
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            print(f"Create checkpoint_dir successfully.")

        # nếu có train trước rồi và load file checkpoint lên để train tiếp
        if os.path.exists(BEST_CHECKPOINT_PATH):
            print(
                f"Checkpoint found at [{BEST_CHECKPOINT_PATH}].\nLoading [{BEST_CHECKPOINT_PATH}]..."
            )
            model, optimizer, start_epoch, checkpoint_mAP, checkpoint_loss = load_checkpoint(
                BEST_CHECKPOINT_PATH,
                model,
                optimizer,
                device=DEVICE,
                load_weights_only=False,
            )

            # Gán giá trị best_mAP từ checkpoint làm checkpoint_mAP
            best_mAP = checkpoint_mAP
            print(f"Loaded best mAP from checkpoint: {best_mAP:.4f}")
            best_loss = checkpoint_loss
            print(f"Loaded best loss from checkpoint: {best_loss:.4f}")

        # chưa train lần nào thì sẽ tạo checkpoint và bắt đầu train từ đầu
        else:
            print(f"Checkpoint not found at [{BEST_CHECKPOINT_PATH}]. \nCreating new checkpoint...")
            
            # Nếu không có checkpoint, bắt đầu từ 0
            best_mAP = 0.0  
            best_loss = float('inf')
            save_checkpoint(
                model,
                optimizer,
                start_epoch,
                best_mAP,
                best_loss,
                filepath=BEST_CHECKPOINT_PATH,
                weights_only=False,
            )
            print(f"Saved new checkpoint at [{BEST_CHECKPOINT_PATH}]")
    else:
        best_mAP = 0.0
        best_loss = float('inf')

    # Training and validation loop
    for epoch in range(start_epoch, EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]:")

        # Train model
        train_loss = train_model(model, train_loader, criterion, optimizer, DEVICE)

        # Validate model and mAP
        val_loss, mAP_score = validate_model(model, val_loader, criterion, DEVICE)

        print(f"Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f} | mAP: {mAP_score:.4f}\n")

        # Lưu checkpoint chỉ khi LOAD_MODEL=True
        if LOAD_MODEL:

            # Save checkpoint if mAP improves
            if mAP_score > best_mAP or train_loss < best_loss:
                print(f"New best mAP: {mAP_score:.4f} (Previous best mAP: {best_mAP:.4f})")
                print(f"New best loss: {train_loss:.4f} (Previous best loss: {best_loss:.4f})")
                
                best_mAP = mAP_score
                best_loss = train_loss
                
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    best_mAP,
                    best_loss,
                    filepath=BEST_CHECKPOINT_PATH,
                    weights_only=False,
                )
                print(f"Saved best checkpoint with mAP: {best_mAP:.4f} and new loss: {best_loss:.4f}")

            # Save periodic checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                periodic_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch + 1}.pth")
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    mAP_score,
                    val_loss,
                    filepath=periodic_checkpoint_path,
                    weights_only=False,
                )
                print(f"Saved periodic checkpoint at {periodic_checkpoint_path}")
                print(f"Saved checkpoint at {epoch + 1} epoch with:\n(mAP : {mAP_score:.4f}) - (val_loss : {val_loss:.4f})")

    print("\nTraining completed!")


if __name__ == "__main__":
    main()
