import torch
from tqdm import tqdm
import os
from loss import compute_iou

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

def save_checkpoint(model, optimizer, epoch, loss, filepath="checkpoint.pth", weights_only=False):
    """
    Save checkpoint to a file.

    Args:
        model (nn.Module): Model instance to save.
        optimizer (torch.optim.Optimizer): Optimizer instance to save.
        epoch (int): Current epoch number.
        loss (float): Current loss value.
        filepath (str): Path to save the checkpoint file.
        weights_only (bool): If True, only save model weights. Default is False.
    """

    try:
        if weights_only:
            checkpoint = {
                "model_state_dict" : model.state_dict(),
            }
        else:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "loss": loss,
            }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved at [{filepath}]")
    except Exception as e:
        print(f"Failed to save checkpoint at [{filepath}]. Error: {e}")


def load_checkpoint(filepath, model, optimizer=None, device="cpu", load_weights_only=False):
    """
    Load checkpoint
    Args:
        filepath (str): Path to the checkpoint file.
        model (nn.Module): Model instance to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer instance to load states into. Default is None.
        device (str or torch.device): Device to load checkpoint to (e.g., "cpu" or "cuda").
        load_weights_only (bool): Whether to load only the model weights. Default is False.

    Returns:
        model (nn.Module): Model with loaded weights.
        optimizer (torch.optim.Optimizer or None): Optimizer with loaded states if load_weights_only=False.
        start_epoch (int): Starting epoch if load_weights_only=False.
        loss (float or None): Loss value if load_weights_only=False.
    """

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at {filepath}")

    print(f"Loading checkpoint from [{filepath}]...")
    checkpoint = torch.load(filepath, map_location=device,) # mặc định sẽ load toàn bộ model

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Model weights loaded.")

    if not load_weights_only: # khi load_weight_only = False thì sẽ load toàn bộ
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Optimizer state loaded.")

        start_epoch = checkpoint.get("epoch", 0)
        loss = checkpoint.get("loss", float("inf"))
        return model, optimizer, start_epoch, loss

    return model, None, 0, None # trả về thông số model khi load_weight_only = True

def mean_average_precision(pred_boxes, truth_boxes, iou_threshold=0.5):
    # phải xử lý lại kích thước pred_boxes, truth_boxes để truyền vào get_iou
    
    pass