
import torch
import torch.nn as nn
import torch.nn.functional as F
def object_detection_loss(pred_boxes, pred_classes, pred_conf, gt_boxes, gt_labels, iou_threshold=0.5):
    """
    Calculate object detection loss similar to typical ResNet-based detectors.
    
    Args:
        pred_boxes: List of tensors with predicted bounding boxes [batch_size, num_boxes, 4]
        pred_classes: List of tensors with predicted class scores [batch_size, num_boxes, num_classes]
        pred_conf: List of tensors with predicted confidence scores [batch_size, num_boxes, 1]
        gt_boxes: List of tensors with ground truth boxes [batch_size, num_gt_boxes, 4]
        gt_labels: List of tensors with ground truth labels [batch_size, num_gt_boxes]
        iou_threshold: IoU threshold for positive matches
    
    Returns:
        tuple: (total_loss, class_loss, box_loss, conf_loss)
    """
    
    batch_size = len(pred_boxes)
    device = pred_boxes[0].device
    
    # Loss weights
    lambda_class = 1.0
    lambda_box = 5.0
    lambda_conf = 1.0
    
    # Initialize loss components
    class_loss = torch.tensor(0.0, device=device)
    box_loss = torch.tensor(0.0, device=device)
    conf_loss = torch.tensor(0.0, device=device)
    
    # Process each item in the batch
    for i in range(batch_size):
        # Skip if no predictions or ground truth
        if len(pred_boxes[i]) == 0 or len(gt_boxes[i]) == 0:
            continue
        
        # Get current batch item predictions and ground truth
        cur_pred_boxes = pred_boxes[i]  # [num_pred, 4]
        cur_pred_classes = pred_classes[i]  # [num_pred, num_classes]
        cur_pred_conf = pred_conf[i]  # [num_pred, 1]
        cur_gt_boxes = gt_boxes[i]  # [num_gt, 4]
        cur_gt_labels = gt_labels[i]  # [num_gt]
        
        num_pred = len(cur_pred_boxes)
        num_gt = len(cur_gt_boxes)
        
        # Calculate IoU between all pred and gt boxes
        ious = box_iou(cur_pred_boxes, cur_gt_boxes)  # [num_pred, num_gt]
        
        # For each gt box, find the best matching pred box
        best_pred_idx_per_gt = torch.argmax(ious, dim=0)  # [num_gt]
        
        # For each pred box, find the best matching gt box
        best_gt_ious, best_gt_idx = torch.max(ious, dim=1)  # [num_pred]
        
        # Create a mask for positive matches (pred boxes matched to gt)
        pos_mask = best_gt_ious >= iou_threshold  # [num_pred]
        
        # CONFIDENCE LOSS
        # Target: 1 for matched boxes, 0 for unmatched
        conf_targets = torch.zeros_like(cur_pred_conf.squeeze(-1))
        conf_targets[pos_mask] = 1.0
        
        # Binary cross entropy for confidence
        conf_loss_i = F.binary_cross_entropy_with_logits(
            cur_pred_conf.squeeze(-1),
            conf_targets,
            reduction='sum'
        )
        
        # CLASS LOSS
        # For positive matches, calculate class loss
        if pos_mask.sum() > 0:
            # For each matched pred box, get the gt class
            matched_gt_classes = cur_gt_labels[best_gt_idx[pos_mask]]
            
            # One-hot encoding of gt classes
            num_classes = cur_pred_classes.size(-1)
            class_targets = torch.zeros((pos_mask.sum(), num_classes), device=device)
            class_targets.scatter_(1, matched_gt_classes.unsqueeze(1), 1)
            
            # Cross entropy for classification
            class_loss_i = F.binary_cross_entropy_with_logits(
                cur_pred_classes[pos_mask],
                class_targets,
                reduction='sum'
            )
            class_loss += class_loss_i
            
            # BOUNDING BOX LOSS
            # L1 loss for bounding box regression (only for positive matches)
            matched_gt_boxes = cur_gt_boxes[best_gt_idx[pos_mask]]
            box_loss_i = F.smooth_l1_loss(
                cur_pred_boxes[pos_mask],
                matched_gt_boxes,
                reduction='sum'
            )
            box_loss += box_loss_i
        
        conf_loss += conf_loss_i
    
    # Normalize losses by batch size
    class_loss /= batch_size
    box_loss /= batch_size
    conf_loss /= batch_size
    
    # Weight and combine losses
    total_loss = lambda_class * class_loss + lambda_box * box_loss + lambda_conf * conf_loss
    
    return total_loss+class_loss+box_loss+conf_loss


def box_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes.
    boxes1 and boxes2 should be in format [x1, y1, x2, y2].
    
    Args:
        boxes1: [N, 4] tensor
        boxes2: [M, 4] tensor
    
    Returns:
        IoU: [N, M] tensor containing IoU values
    """
    N = boxes1.size(0)
    M = boxes2.size(0)
    
    # Get area of boxes1
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N]
    area1 = area1.unsqueeze(1).expand(N, M)  # [N, M]
    
    # Get area of boxes2
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M]
    area2 = area2.unsqueeze(0).expand(N, M)  # [N, M]
    
    # Get coordinates of intersection
    lt = torch.max(
        boxes1[:, :2].unsqueeze(1).expand(N, M, 2),
        boxes2[:, :2].unsqueeze(0).expand(N, M, 2)
    )  # [N, M, 2]
    
    rb = torch.min(
        boxes1[:, 2:].unsqueeze(1).expand(N, M, 2),
        boxes2[:, 2:].unsqueeze(0).expand(N, M, 2)
    )  # [N, M, 2]
    
    # Calculate width and height of intersection
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    
    # Calculate area of intersection
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    # Calculate IoU
    iou = inter / (area1 + area2 - inter)
    
    return iou