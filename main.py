import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import PascalVOC2012Dataset
from loss import YOLOv1Loss
from model import Yolov1

# set device
device = torch.device("cuda" if torch.cuda.is_available else 'cpu')

# Hyperparameters
lr = 2e-5
batch_size = 16
epochs = 10
root_dir = "/Data/VOC2012"

def main():
    pass


if __name__ == '__main__':
    main()