import torch
import torch.nn.functional as F
from pdb import set_trace
def object_detection_loss(pred_boxes, pred_classes, pred_conf, gt_boxes, gt_labels, distance_scale=0.5):
    """
    Compute object detection loss by matching each predicted box to the closest ground truth,
    and assigning confidence dynamically based on match quality.

    Assumes classes are in range 0-86 (91 classes total).
    """
    batch_size = len(pred_boxes)
    device = pred_boxes[0].device if len(pred_boxes) > 0 and len(pred_boxes[0]) > 0 else 'cuda'
    num_classes = 91 # Fixed number of classes (0-79)

    # Loss weights
    lambda_class = 1.0
    lambda_box = 2.0
    lambda_conf = 1.0

    class_loss = torch.tensor(0.0, device=device)
    box_loss = torch.tensor(0.0, device=device)
    conf_loss = torch.tensor(0.0, device=device)

    total_predictions = 0

    for i in range(batch_size):
        if len(pred_boxes[i]) == 0:
            continue  # Skip empty predictions

        cur_pred_boxes = pred_boxes[i].to(device)  # [num_pred, 4]
        cur_pred_classes = pred_classes[i].to(device)  # [num_pred, num_classes]
        cur_pred_conf = pred_conf[i].squeeze(-1).to(device)  # [num_pred]

        total_predictions += len(cur_pred_boxes)

        if len(gt_boxes[i]) == 0:
            # If no ground truth, all predictions are false positives with low confidence
            conf_loss += F.binary_cross_entropy_with_logits(
                cur_pred_conf, torch.zeros_like(cur_pred_conf), reduction='sum'
            )
            continue

        cur_gt_boxes = gt_boxes[i].to(device)  # [num_gt, 4]
        cur_gt_labels = gt_labels[i].to(device)  # [num_gt]

        # Ensure ground truth labels are within valid range
        if cur_gt_labels.min() < 0 or cur_gt_labels.max() >= num_classes:
            print(f"Warning: Invalid ground truth class indices detected. Min: {cur_gt_labels.min()}, Max: {cur_gt_labels.max()}")
            # Clamp to valid range (0-79)
            cur_gt_labels = torch.clamp(cur_gt_labels, 0, num_classes - 1)

        # Compute center coordinates
        pred_centers = (cur_pred_boxes[:, :2] + cur_pred_boxes[:, 2:]) / 2  # [num_pred, 2]
        gt_centers = (cur_gt_boxes[:, :2] + cur_gt_boxes[:, 2:]) / 2  # [num_gt, 2]

        # Compute Euclidean distance matrix
        dists = torch.cdist(pred_centers, gt_centers, p=2)  # [num_pred, num_gt]

        # Match each prediction to the closest ground truth
        best_gt_dist, best_gt_idx = torch.min(dists, dim=1)  # [num_pred]

        # Get matched GT values
        matched_gt_boxes = cur_gt_boxes[best_gt_idx]  # [num_pred, 4]
        matched_gt_classes = cur_gt_labels[best_gt_idx]  # [num_pred]

        # Double-check matched classes are in valid range
        matched_gt_classes = torch.clamp(matched_gt_classes, 0, num_classes - 1)

        # Compute box loss for all matched pairs
        box_loss_per_pred = F.smooth_l1_loss(cur_pred_boxes, matched_gt_boxes, reduction='none').mean(dim=1)
        box_loss += box_loss_per_pred.sum()


        # Compute class loss using standard cross entropy
        class_loss_per_pred = F.cross_entropy(
            cur_pred_classes, matched_gt_classes.long(), reduction='none'
        )
        class_loss += class_loss_per_pred.sum()

        # Compute class correctness for confidence target
        pred_class_labels = torch.argmax(cur_pred_classes, dim=1)  # Get predicted labels
        class_correct = (pred_class_labels == matched_gt_classes).float()  # 1 if correct, 0 otherwise

        # Compute confidence target dynamically
        conf_targets = torch.exp(-distance_scale * best_gt_dist) * (1 - box_loss_per_pred) * class_correct

        # Confidence loss
        conf_loss += F.binary_cross_entropy_with_logits(
            cur_pred_conf, conf_targets, reduction='sum'
        )

    # Normalize by the number of predictions to prevent exploding loss
    total_predictions = max(total_predictions, 1)  # Avoid division by zero

    class_loss /= total_predictions
    box_loss /= total_predictions
    conf_loss /= total_predictions

    # Weighted sum of losses
    total_loss = lambda_class * class_loss + lambda_box * box_loss + lambda_conf * conf_loss


    return total_loss, class_loss, box_loss, conf_loss


def custom_cross_entropy(pred_logits, target_classes, num_classes=91):
    """
    Compute cross-entropy loss manually.
    pred_logits: [num_pred, num_classes] raw class scores (logits)
    target_classes: [num_pred] ground truth class indices
    num_classes: Fixed at 80 (classes 0-79)
    """
    # Ensure target_classes is long
    target_classes = target_classes.long()

    # Clamp indices to valid range (0-79)
    target_classes = torch.clamp(target_classes, 0, num_classes - 1)

    pred_probs = torch.softmax(pred_logits, dim=1)  # Step 1: Softmax
    log_probs = torch.log(pred_probs + 1e-9)  # Step 2: Log probabilities

    # Safe indexing
    target_log_probs = log_probs[torch.arange(len(target_classes)), target_classes]  # Step 3

    loss = -target_log_probs.mean()  # Step 4
    return loss