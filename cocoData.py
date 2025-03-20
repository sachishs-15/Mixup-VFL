import torch
import os
import cv2

def load_coco_data(dataset_path="datasets/coco", target_size=(480, 640)):
    """
    Load and combine both train and val splits while maintaining:
    - Consistent tensor dimensions
    - Proper bounding box transformations
    - Batch-compatible formats
    """
    def load_split(split):
        X, y = [], []
        image_dir = os.path.join(dataset_path, "images", split)
        label_dir = os.path.join(dataset_path, "labels", split)

        for img_file in [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]:
            # Image loading and processing
            img_path = os.path.join(image_dir, img_file)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Preserve aspect ratio with letterbox resize
            h, w = image.shape[:2]
            scale = min(target_size[1]/w, target_size[0]/h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            dw = target_size[1] - new_w
            dh = target_size[0] - new_h
            
            # Center padding
            top, bottom = dh//2, dh - (dh//2)
            left, right = dw//2, dw - (dw//2)
            padded = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                      cv2.BORDER_CONSTANT, value=(114,114,114))
            
            # Label processing with correct scaling
            label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")
            boxes = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        class_id, xc, yc, bw, bh = map(float, line.strip().split())
                        
                        # Adjust for letterbox scaling and padding
                        xc = (xc * w * scale + left) / target_size[1]
                        yc = (yc * h * scale + top) / target_size[0]
                        bw = (bw * w * scale) / target_size[1]
                        bh = (bh * h * scale) / target_size[0]
                        
                        boxes.append([class_id, xc, yc, bw, bh])

            # Convert to PyTorch tensors
            X.append(torch.from_numpy(padded).permute(2,0,1).float() / 255.0)  # CHW format
            y.append(torch.tensor(boxes) if boxes else torch.zeros((0,5)))
            
        return X, y

    # Load and combine both splits
    train_X, train_y = load_split('train')
    val_X, val_y = load_split('val')
    
    return train_X ,train_y,val_X,val_y,target_size[1]

