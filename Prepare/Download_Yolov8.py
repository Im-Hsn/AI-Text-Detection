import os
import requests

# Define the YOLOv8 Medium model URL
YOLOV8_MEDIUM_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt"

# Set the target directory
TARGET_DIR = "./Yolov8"

# Create the directory if it doesn't exist
os.makedirs(TARGET_DIR, exist_ok=True)

# Define the file path to save the model
model_path = os.path.join(TARGET_DIR, "yolov8m.pt")

# Check if the file already exists
if not os.path.exists(model_path):
    print("Downloading YOLOv8 Medium model...")
    response = requests.get(YOLOV8_MEDIUM_URL, stream=True)
    if response.status_code == 200:
        # Save the file
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"YOLOv8 Medium model downloaded successfully and saved to {model_path}.")
    else:
        print(f"Failed to download YOLOv8 Medium model. HTTP Status Code: {response.status_code}")
else:
    print(f"YOLOv8 Medium model already exists at {model_path}.")
