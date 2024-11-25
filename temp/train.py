import torch
import torch.optim as optim
import torchvision.transforms as transforms
from dataset import PascalVOC2012Dataset
from loss import YOLOv1Loss

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            if isinstance(t, transforms.Resize):
                old_width, old_height = img.size
                img = t(img)
                new_width, new_height = img.size
                # Scale bounding boxes to match the resized image
                bboxes = [[box[0] * new_width / old_width, 
                           box[1] * new_height / old_height, 
                           box[2] * new_width / old_width, 
                           box[3] * new_height / old_height, box[4]] for box in bboxes]
            else:
                img = t(img)
                
        return img, bboxes
    
transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])

def train_fn(train_loader, model, optimizer, loss_fn, device):
    model.train()
    mean_loss = []
    for batch_idx, (images, targets) in enumerate(train_loader):
        print(f"Processing batch {batch_idx + 1}/{len(train_loader)}")
        images = images.to(device)
        targets = targets['yolo_targets'].to(device)

        # Forward pass
        predictions = model(images)

        # Đảm bảo predictions và targets có cùng kích thước
        assert predictions.shape == targets.shape, f"Shape mismatch: {predictions.shape} vs {targets.shape}"

        # Compute the loss
        loss = loss_fn(predictions, targets)
        mean_loss.append(loss.item())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Mean loss for epoch: {sum(mean_loss) / len(mean_loss)}")
