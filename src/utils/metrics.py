import torch
import torch.nn.functional as F
import numpy as np
import os
from typing import Dict, Optional, Union, List

def calculate_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: Optional[int] = None,
    ignore_index: Optional[int] = 255
) -> torch.Tensor:
    """
    Calculate confusion matrix.
    
    Args:
        predictions: Predicted labels (B, H, W)
        targets: Ground truth labels (B, H, W)
        num_classes: Number of classes
        ignore_index: Index to ignore
        
    Returns:
        Confusion matrix (C, C)
    """
    if num_classes is None:
        num_classes = max(predictions.max().item(), targets.max().item()) + 1
    
    mask = (targets != ignore_index) if ignore_index is not None else None
    
    confusion_matrix = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.long,
        device=predictions.device
    )
    
    if mask is not None:
        predictions = predictions[mask]
        targets = targets[mask]
    
    for i in range(num_classes):
        for j in range(num_classes):
            confusion_matrix[i, j] = torch.sum(
                (predictions == i) & (targets == j)
            )
    
    return confusion_matrix

def calculate_iou(
    confusion_matrix: torch.Tensor,
    smooth: float = 1e-6
) -> torch.Tensor:
    """
    Calculate IoU for each class from confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix (C, C)
        smooth: Smoothing factor
        
    Returns:
        IoU for each class (C,)
    """
    intersection = torch.diag(confusion_matrix)
    union = (
        confusion_matrix.sum(dim=0) +  # Ground truth
        confusion_matrix.sum(dim=1) -   # Prediction
        intersection
    )
    iou = (intersection + smooth) / (union + smooth)
    return iou

def calculate_dice(
    confusion_matrix: torch.Tensor,
    smooth: float = 1e-6
) -> torch.Tensor:
    """
    Calculate Dice coefficient for each class from confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix (C, C)
        smooth: Smoothing factor
        
    Returns:
        Dice coefficient for each class (C,)
    """
    intersection = torch.diag(confusion_matrix)
    sum_predictions = confusion_matrix.sum(dim=0)
    sum_targets = confusion_matrix.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (sum_predictions + sum_targets + smooth)
    return dice

def calculate_precision_recall(
    confusion_matrix: torch.Tensor,
    smooth: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate precision and recall for each class from confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix (C, C)
        smooth: Smoothing factor
        
    Returns:
        Tuple of (precision, recall) for each class (C,)
    """
    true_positives = torch.diag(confusion_matrix)
    sum_predictions = confusion_matrix.sum(dim=0)
    sum_targets = confusion_matrix.sum(dim=1)
    
    precision = (true_positives + smooth) / (sum_predictions + smooth)
    recall = (true_positives + smooth) / (sum_targets + smooth)
    
    return precision, recall

def calculate_f1_score(
    precision: torch.Tensor,
    recall: torch.Tensor,
    smooth: float = 1e-6
) -> torch.Tensor:
    """
    Calculate F1 score for each class from precision and recall.
    
    Args:
        precision: Precision for each class (C,)
        recall: Recall for each class (C,)
        smooth: Smoothing factor
        
    Returns:
        F1 score for each class (C,)
    """
    f1 = (2.0 * precision * recall + smooth) / (precision + recall + smooth)
    return f1

def calculate_accuracy(
    confusion_matrix: torch.Tensor,
    smooth: float = 1e-6
) -> float:
    """
    Calculate overall pixel accuracy from confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix (C, C)
        smooth: Smoothing factor
        
    Returns:
        Overall pixel accuracy
    """
    correct = torch.diag(confusion_matrix).sum()
    total = confusion_matrix.sum()
    accuracy = (correct + smooth) / (total + smooth)
    return accuracy.item()

def calculate_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255
) -> Dict[str, torch.Tensor]:
    """
    Calculate segmentation metrics.
    
    Args:
        predictions: Predicted labels [B, H, W]
        targets: Ground truth labels [B, H, W]
        num_classes: Number of classes
        ignore_index: Index to ignore
        
    Returns:
        Dictionary of metrics
    """
    # Create mask for valid pixels
    valid_mask = targets != ignore_index
    
    # Calculate per-class IoU
    class_ious = []
    for class_idx in range(num_classes):
        pred_mask = (predictions == class_idx) & valid_mask
        target_mask = (targets == class_idx) & valid_mask
        
        intersection = torch.sum(pred_mask & target_mask).float()
        union = torch.sum(pred_mask | target_mask).float()
        
        iou = (intersection + 1e-6) / (union + 1e-6)
        class_ious.append(iou)
    
    # Calculate mean IoU
    mean_iou = torch.mean(torch.stack(class_ious))
    
    # Calculate per-class Dice
    class_dices = []
    for class_idx in range(num_classes):
        pred_mask = (predictions == class_idx) & valid_mask
        target_mask = (targets == class_idx) & valid_mask
        
        intersection = torch.sum(pred_mask & target_mask).float()
        total = torch.sum(pred_mask).float() + torch.sum(target_mask).float()
        
        dice = (2.0 * intersection + 1e-6) / (total + 1e-6)
        class_dices.append(dice)
    
    # Calculate mean Dice
    mean_dice = torch.mean(torch.stack(class_dices))
    
    # Calculate accuracy
    correct = torch.sum((predictions == targets) & valid_mask).float()
    total = torch.sum(valid_mask).float()
    accuracy = (correct + 1e-6) / (total + 1e-6)
    
    return {
        'iou': mean_iou,
        'dice': mean_dice,
        'accuracy': accuracy,
        'class_iou': torch.stack(class_ious),
        'class_dice': torch.stack(class_dices)
    }

def calculate_boundary_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    kernel_size: int = 3,
    smooth: float = 1e-6
) -> torch.Tensor:
    """
    Calculate boundary IoU for evaluating segmentation boundaries.
    
    Args:
        predictions: Predicted labels (B, H, W)
        targets: Ground truth labels (B, H, W)
        kernel_size: Size of dilation kernel
        smooth: Smoothing factor
        
    Returns:
        Boundary IoU
    """
    # Extract boundaries
    pred_boundaries = extract_boundaries(predictions, kernel_size)
    target_boundaries = extract_boundaries(targets, kernel_size)
    
    # Calculate IoU
    intersection = torch.sum(pred_boundaries & target_boundaries)
    union = torch.sum(pred_boundaries | target_boundaries)
    
    boundary_iou = (intersection + smooth) / (union + smooth)
    return boundary_iou

def extract_boundaries(
    masks: torch.Tensor,
    kernel_size: int = 3
) -> torch.Tensor:
    """
    Extract boundaries from segmentation masks.
    
    Args:
        masks: Input masks (B, H, W)
        kernel_size: Size of dilation kernel
        
    Returns:
        Binary boundary masks
    """
    # Create dilation kernel
    kernel = torch.ones(
        (1, 1, kernel_size, kernel_size),
        device=masks.device
    )
    
    # Add channel dimension
    masks = masks.unsqueeze(1).float()
    
    # Dilate masks
    dilated = F.conv2d(
        masks,
        kernel,
        padding=kernel_size // 2
    )
    
    # Get boundaries
    boundaries = (dilated > 0) & (dilated < kernel_size * kernel_size)
    return boundaries.squeeze(1)

def save_predictions(predictions: Dict[str, Union[np.ndarray, torch.Tensor]],
                    output_dir: str) -> None:
    """
    Save predictions to disk.
    
    Args:
        predictions: Dictionary containing predictions
        output_dir: Directory to save predictions to
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert torch tensors to numpy if needed
    processed_predictions = {}
    for k, v in predictions.items():
        if isinstance(v, torch.Tensor):
            processed_predictions[k] = v.cpu().numpy()
        else:
            processed_predictions[k] = v
    
    # Save as compressed npz
    output_file = os.path.join(output_dir, 'predictions.npz')
    np.savez_compressed(output_file, **processed_predictions) 