import torch
import os
import cv2
import json
import numpy as np
from torchvision import transforms
from PIL import Image
from pdb import set_trace

def load_coco_data(dataset_path="datasets", target_size=(224, 224), num_train=100, num_val=50):
    """
    Load a specified number of train and val images while ensuring:
    - COCO JSON annotation parsing
    - Consistent tensor dimensions
    - Proper bounding box transformations
    - Batch-compatible formats
    - Image preprocessing with normalization

    :param dataset_path: Path to COCO dataset root
    :param target_size: Target image size (height, width)
    :param num_train: Number of training images to load
    :param num_val: Number of validation images to load
    :return: train_X, train_y, val_X, val_y, target_size[1]
    """
    
    # Define preprocessing transformation based on target size
    # Since we're using a wider format (224x448), we need to adjust the preprocessing
    resize_size = (int(256 * (target_size[1] / 224)), int(256 * (target_size[0] / 224)))  # Proportional resize
    
    preprocess = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def load_split(split, num_samples):
        X, y = [], []
        image_dir = os.path.join(dataset_path, split)
        annotation_path = os.path.join(dataset_path, "annotations", f"instances_{split}.json")

        # Load COCO annotations
        with open(annotation_path, "r") as f:
            coco_data = json.load(f)

        # Mapping from image_id to file_name
        image_id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}
        
        # Mapping from image_id to bounding boxes
        image_id_to_boxes = {}
        for ann in coco_data["annotations"]:
            image_id = ann["image_id"]
            bbox = ann["bbox"]  # [x_min, y_min, width, height]
            class_id = ann["category_id"]  # COCO class ID

            if image_id not in image_id_to_boxes:
                image_id_to_boxes[image_id] = []
            image_id_to_boxes[image_id].append([class_id] + bbox)

        # Select only the specified number of images
        selected_image_ids = list(image_id_to_filename.keys())[:num_samples]

        for image_id in selected_image_ids:
            img_file = image_id_to_filename[image_id]
            img_path = os.path.join(image_dir, img_file)

            # Read image
            
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image for torchvision transforms
            pil_image = Image.fromarray(image)
            
            # Apply preprocessing with normalization
            input_tensor = preprocess(pil_image)
            
            # Store original dimensions for bounding box transformation
            orig_h, orig_w = image.shape[:2]

            # Process bounding boxes
            boxes = []
            if image_id in image_id_to_boxes:
                for box in image_id_to_boxes[image_id]:
                    class_id, x_min, y_min, bw, bh = box

                    # Calculate scale factors based on new resize dimensions
                    resize_w, resize_h = resize_size
                    scale_w = resize_w / orig_w
                    scale_h = resize_h / orig_h
                    scale = min(scale_w, scale_h)
                    
                    # Calculate new dimensions after preserving aspect ratio
                    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
                    
                    # Calculate padding for center crop
                    left_pad = (resize_w - new_w) // 2
                    top_pad = (resize_h - new_h) // 2
                    
                    # Center crop offset
                    crop_left = (resize_w - target_size[1]) // 2
                    crop_top = (resize_h - target_size[0]) // 2
                    
                    # Transform box coordinates: scale + pad + crop
                    x_center = (x_min + bw / 2) * scale + left_pad - crop_left
                    y_center = (y_min + bh / 2) * scale + top_pad - crop_top
                    scaled_bw = bw * scale
                    scaled_bh = bh * scale
                    
                    # Check if box is within the cropped area
                    if (x_center >= 0 and x_center < target_size[1] and 
                        y_center >= 0 and y_center < target_size[0]):
                        
                        # Normalize to [0,1] for CNN compatibility
                        x_center /= target_size[1]  # normalize by width (448)
                        y_center /= target_size[0]  # normalize by height (224)
                        scaled_bw /= target_size[1]
                        scaled_bh /= target_size[0]
                        
                        boxes.append([class_id, x_center, y_center, scaled_bw, scaled_bh])

            # Add to dataset
            X.append(input_tensor)  # Already in CHW format from transforms.ToTensor()
            y.append(torch.tensor(boxes) if boxes else torch.zeros((0, 5)))
        
        return X, y

    # Load train and val data with specified numbers
    train_X, train_y = load_split('train2017', num_train)
    val_X, val_y = load_split('val2017', num_val)
    
    return train_X, train_y, val_X, val_y, target_size[1]