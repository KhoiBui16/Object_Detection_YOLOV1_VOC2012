import torch
from tqdm import tqdm
import os
from loss import compute_iou

def convert_to_boxes(tensor, is_prediction=True):
    """
    Convert tensor predictions/targets to bounding box format.

    Args:
        tensor (Tensor): Predictions or targets [B, S, S, 5 + C].
        is_prediction (bool): Whether the input is predictions or targets.

    Returns:
        list: List of bounding boxes in the format
            [x_center, y_center, width, height, conf, class_probs...] for predictions
            [x_center, y_center, width, height, label] for targets.
    """
    B, S, S, _ = tensor.shape
    boxes = []
    tensor = tensor.cpu()

    for b in range(B):
        for i in range(S):
            for j in range(S):
                cell = tensor[b, i, j]
                if is_prediction:
                    for k in range(2):  # B (number of bounding boxes per cell)
                        conf = cell[4 + k * 5]
                        if conf > 0.5:  # Confidence threshold
                            x_center = (cell[k * 5] + j) / S
                            y_center = (cell[k * 5 + 1] + i) / S
                            width = cell[k * 5 + 2]
                            height = cell[k * 5 + 3]
                            class_probs = cell[10:]
                            boxes.append([x_center, y_center, width, height, conf, *class_probs])
                else:
                    if cell[4] > 0:  # Objectness score > 0
                        x_center = (cell[0] + j) / S
                        y_center = (cell[1] + i) / S
                        width = cell[2]
                        height = cell[3]
                        label = torch.argmax(cell[5:]).item()
                        boxes.append([x_center, y_center, width, height, label])

    return boxes


def train_model(model, dataloader, criterion, optimizer, device):
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
        desc='Training',
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
    all_pred_boxes = []
    all_truth_boxes = []

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

            # Chuyển đổi predictions và targets thành danh sách bounding box
            batch_pred_boxes = convert_to_boxes(predictions)
            batch_truth_boxes = convert_to_boxes(targets, is_prediction=False)

            all_pred_boxes.extend(batch_pred_boxes)
            all_truth_boxes.extend(batch_truth_boxes)

            progress_bar.set_postfix({'Loss': loss.item()})

        mAP_score = mean_average_precision(
            all_pred_boxes,
            all_truth_boxes
        )

    # Return average validation loss and mAP
    return total_loss / len(dataloader), mAP_score

def save_checkpoint(model, optimizer, epoch, mAP, loss, filepath="checkpoint.pth", weights_only=False):
    """
    Save checkpoint to a file.

    Args:
        model (nn.Module): Model instance to save.
        optimizer (torch.optim.Optimizer): Optimizer instance to save.
        epoch (int): Current epoch number.
        loss (float): Current loss value.
        mAP (float): Current mAP value.
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
                "mAP": mAP,
                "loss" : loss
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
        mAP (float or None): mAP value if load_weights_only=False.
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
        mAP = checkpoint.get("mAP", 0.0)
        loss = checkpoint.get("loss", float('inf'))
        return model, optimizer, start_epoch, mAP, loss

    return model, None, 0, None, None # trả về thông số model khi load_weight_only = True

def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, num_classes=20):
    """
    Calculate mean average precision.

    Parameters:
        pred_boxes (list):  Predictions with each element formatted as [x_center, y_center, w, h, conf, class_probs...]
                            Shape: [num_pred_boxes, 5 + num_classes]
        true_boxes (list):  Ground truths with each element formatted as [x_center, y_center, w, h, label]
                            Shape: [num_true_boxes, 5]
        iou_threshold (float): IOU threshold to count a detection as a true positive.
        num_classes (int): Number of classes.

    Returns:
        float: Mean average precision across all classes.
    """
    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        # Filter for predictions and ground truths of class c
        detections = [
            box for box in pred_boxes
            if torch.argmax(torch.tensor(box[5:])).item()-4 == c  # Shape: [5 + num_classes]
        ]

        ground_truths = [
            box for box in true_boxes
            if box[4]-4 == c  # Shape: [5]
        ]

        # Move all detections and ground truths to the same device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        detections = [torch.tensor(box).to(device) for box in detections]
        ground_truths = [torch.tensor(box).to(device) for box in ground_truths]

        # Tổng số ground truth của class hiện tại
        total_true_bboxes = len(ground_truths)
        matched = [False] * total_true_bboxes

        if len(detections) == 0 or len(ground_truths) == 0:
            average_precisions.append(0)
            continue

        # Compute IoU matrix for all detections and ground truths
        iou_matrix = torch.zeros((len(detections), len(ground_truths)), device=device)
        for i, detection in enumerate(detections):
            for j, gt in enumerate(ground_truths):
                iou_matrix[i, j] = compute_iou(
                    detection[:4].unsqueeze(0),  # Shape: [1, 4]
                    gt[:4].unsqueeze(0),        # Shape: [1, 4]
                    split_size=7,
                    batch=1
                ).max().item()  # Lấy giá trị lớn nhất nếu hàm trả về tensor


        # Sắp xếp các dự đoán theo confidence giảm dần
        detections.sort(key=lambda x: x[4], reverse=True)

        TP = torch.zeros(len(detections), device=device)  # Shape: [num_detections]
        FP = torch.zeros(len(detections), device=device)  # Shape: [num_detections]

        # Xét từng bounding box dự đoán
        for detection_idx, detection in enumerate(detections):
            if iou_matrix[detection_idx].max() > iou_threshold:
                best_gt_idx = iou_matrix[detection_idx].argmax().item()
                if not matched[best_gt_idx]:
                    TP[detection_idx] = 1
                    matched[best_gt_idx] = True
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        # Tính cumulative TP và FP
        TP_cumsum = torch.cumsum(TP, dim=0)  # Shape: [num_detections]
        FP_cumsum = torch.cumsum(FP, dim=0)  # Shape: [num_detections]

        # Tính precision và recall
        recalls = TP_cumsum / (total_true_bboxes + epsilon)  # Shape: [num_detections]
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)  # Shape: [num_detections]

        # Thêm giá trị 1 ở đầu precision và 0 ở đầu recall
        precisions = torch.cat((torch.tensor([1], device=device), precisions))  # Shape: [num_detections + 1]
        recalls = torch.cat((torch.tensor([0], device=device), recalls))  # Shape: [num_detections + 1]

        # Tính Average Precision (AP) bằng diện tích dưới đường P-R
        average_precision = torch.trapz(precisions, recalls)  # Scalar
        average_precisions.append(average_precision)

    # Tính mean Average Precision (mAP)
    return sum(average_precisions) / len(average_precisions)  # Scalar
