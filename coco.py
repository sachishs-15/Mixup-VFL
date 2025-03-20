import yaml
import os
import requests
import zipfile
import cv2
import matplotlib.pyplot as plt

# Load YAML file
yaml_path = "./config/coco.yaml"
with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)

# Get dataset path and download URL
dataset_path = config["path"]
download_url = config["download"]

# Ensure dataset directory exists
os.makedirs(dataset_path, exist_ok=True)

# Define zip file path
zip_path = os.path.join(dataset_path, "coco.zip")

# Check if dataset is already downloaded
if not os.listdir(dataset_path):  # Check if directory is empty
    print(f"Downloading COCO8 dataset from {download_url}...")
    
    response = requests.get(download_url, stream=True)
    response.raise_for_status()  # Ensure successful download
    
    # Save the ZIP file
    with open(zip_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)
    
    print("Download complete. Extracting files...")

    # Extract dataset
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
         zip_ref.extractall(os.path.dirname(dataset_path)) 

    # Remove zip file after extraction
    os.remove(zip_path)
    print(f"✅ COCO8 dataset is ready at {dataset_path}")
else:
    print(f"✅ Dataset already exists at {dataset_path}, skipping download.")

# Define image directory path
image_dir = os.path.join(dataset_path, "images/train")  # Change to 'val' if needed
print("Checking extracted files...")
print(os.listdir(dataset_path))  # Should contain 'images/' and other files
import os

dataset_path = "../datasets/coco8"
print("Extracted contents:", os.listdir(dataset_path))
