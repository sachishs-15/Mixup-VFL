import torch
import os
import cv2
import json
import numpy as np
from torchvision import transforms
from PIL import Image

def load_pascal_data(dataset_path="/kaggle/input/pascal-voc-2012/VOC2012", target_size=(224, 224), num_train=100, num_val=50):
    """
    Load a specified number of train and val images while ensuring:
    - PASCAL VOC XML annotation parsing
    - Consistent tensor dimensions
    - Proper bounding box transformations
    - Batch-compatible formats
    - Image preprocessing with normalization

    :param dataset_path: Path to PASCAL VOC dataset root
    :param target_size: Target image size (height, width)
    :param num_train: Number of training images to load
    :param num_val: Number of validation images to load
    :return: train_X, train_y, val_X, val_y, target_size[1]
    """

    # Define preprocessing transformation
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def parse_annotations(annotation_dir):
        """Parse PASCAL VOC XML annotations and return image bounding boxes with class labels."""
        from xml.etree import ElementTree as ET

        annotations = {}
        class_map = {}  # Map class names to numerical labels
        class_counter = 0

        for xml_file in sorted(os.listdir(annotation_dir)):
            if not xml_file.endswith(".xml"):
                continue

            tree = ET.parse(os.path.join(annotation_dir, xml_file))
            root = tree.getroot()
            image_id = root.find("filename").text
            width = int(root.find("size/width").text)
            height = int(root.find("size/height").text)

            boxes = []
            for obj in root.findall("object"):
                class_name = obj.find("name").text
                if class_name not in class_map:
                    class_map[class_name] = class_counter
                    class_counter += 1

                class_id = class_map[class_name]
                bbox = obj.find("bndbox")
                bbox = obj.find("bndbox")
                x_min = int(float(bbox.find("xmin").text))
                y_min = int(float(bbox.find("ymin").text))
                x_max = int(float(bbox.find("xmax").text))
                y_max = int(float(bbox.find("ymax").text))

                # Convert to COCO-style format [class_id, x_center, y_center, width, height]
                x_center = (x_min + x_max) / 2 / width
                y_center = (y_min + y_max) / 2 / height
                box_w = (x_max - x_min) / width
                box_h = (y_max - y_min) / height

                boxes.append([class_id, x_center, y_center, box_w, box_h])

            annotations[image_id] = boxes

        return annotations, class_map

    def load_split(images_dir, annotations, selected_filenames):
        """Load images and their corresponding bounding boxes."""
        X, y = [], []

        for img_file in selected_filenames:
            img_path = os.path.join(images_dir, img_file)
            if not os.path.exists(img_path):
                continue

            # Read image
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image for torchvision transforms
            pil_image = Image.fromarray(image)

            # Apply preprocessing with normalization
            input_tensor = preprocess(pil_image)

            # Get bounding boxes
            boxes = annotations.get(img_file, [])

            # Add to dataset
            X.append(input_tensor)  # Already in CHW format from transforms.ToTensor()
            y.append(torch.tensor(boxes) if boxes else torch.zeros((0, 5)))

        return X, y

    # Load annotations
    annotation_dir = os.path.join(dataset_path, "Annotations")
    images_dir = os.path.join(dataset_path, "JPEGImages")
    annotations, class_map = parse_annotations(annotation_dir)

    # Get sorted image filenames
    image_filenames = sorted(list(annotations.keys()))

    # Select non-overlapping train and val sets
    train_filenames = image_filenames[:num_train]
    val_filenames = image_filenames[num_train:num_train + num_val]

    # Load train and val data
    train_X, train_y = load_split(images_dir, annotations, train_filenames)
    val_X, val_y = load_split(images_dir, annotations, val_filenames)

    return train_X, train_y, val_X, val_y, target_size[1]