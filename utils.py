import torch
from tqdm import tqdm

def train_model(model, dataloader, criterion, optimizer, device, epoch):
    """
    Training function for one epoch

    Args:
        model (nn.Module): YOLOv1 model
        dataloader (DataLoader): Training data loader
        criterion (nn.Module): Loss function
        optimizer (torch.optim): Optimizer
        device (torch.device): Computing device
        epoch (int): Current epoch number

    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(
        dataloader,
        desc=f'Epoch {epoch + 1} Training',
        unit='batch'
    )
    
    
    for batch_idx,(_, images, targets) in enumerate(progress_bar):
        # print(f"Processing train - batch [{batch_idx + 1}/{len(dataloader)}]")
        # Move to device
        images = images.to(device)
        targets = targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(images)

        # Compute loss
        loss = criterion(predictions, targets)

        # Backward pass
        loss.backward()

        # Optimize
        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        progress_bar.set_postfix({'Loss': loss.item()})

    # Return average loss for the epoch
    return total_loss / len(dataloader)


def validate_model(model, dataloader, criterion, device):
    """
    Validation function

    Args:
        model (nn.Module): YOLOv1 model
        dataloader (DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        device (torch.device): Computing device

    Returns:
        float: Average validation loss
    """
    model.eval()
    total_loss = 0.0

    # Disable gradient computation for validation
    with torch.no_grad():
        progress_bar = tqdm(
            dataloader,
            desc='Validation',
            unit='batch'
        )
        
        for batch_idx,(_,images, targets) in enumerate(progress_bar):
            # print(f"Processin eval batch [{batch_idx + 1}/{len(dataloader)}]")
            # Move to device
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            predictions = model(images)

            # Compute loss
            loss = criterion(predictions, targets)

            # Update metrics
            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': loss.item()})

    # Return average validation loss
    return total_loss / len(dataloader)

def save_checkpoint(model, optimizer, epoch, loss, filepath="checkpoint.pth"):
    """
    Save checkpoint
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at {filepath}")

def load_checkpoint(filepath, model, optimizer=None):
    """
    Load checkpoint
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model weight loaded from {filepath}")

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Optimizer state loaded")

    start_epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", None)
    return model, optimizer, start_epoch, loss