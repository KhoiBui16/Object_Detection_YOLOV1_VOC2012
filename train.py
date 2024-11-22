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
            img, bboxes = t(img), bboxes

        return img, bboxes
    
transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])

def train_fn(train_loader, model, optimizer, loss_fn, device):
    model.train()
    mean_loss = []
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets['yolo_targets'].to(device)

        # Forward pass
        predictions = model(images)
        loss = loss_fn(predictions, targets)
        mean_loss.append(loss.item())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Mean loss for epoch: {sum(mean_loss) / len(mean_loss)}")