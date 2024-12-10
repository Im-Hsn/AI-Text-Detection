import os
import random
import torch
import shutil
from ultralytics import YOLO

def check_pytorch_gpu():
    """Check if PyTorch can access the GPU"""
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is not available. Please ensure PyTorch is installed with CUDA support.")
    device_count = torch.cuda.device_count()
    print(f"PyTorch found {device_count} GPU(s):")
    for i in range(device_count):
        print(f"  Device {i}: {torch.cuda.get_device_name(i)}")

def clear_cuda_cache():
    """Clear CUDA cache to free up memory"""
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print("Cleared CUDA cache.")
    
# def validate_and_fix_labels(label_path):
#     """
#     Validate and fix label coordinates to ensure they are normalized between 0 and 1.
#     """
#     try:
#         if not os.path.exists(label_path):
#             return False

#         with open(label_path, 'r') as f:
#             lines = f.readlines()
        
#         fixed_lines = []
#         modified = False
#         invalid_lines = 0
#         seen_labels = set()
        
#         for line_num, line in enumerate(lines, 1):
#             try:
#                 values = line.strip().split()
#                 if len(values) != 5:
#                     print(f"Warning: Skipping invalid line {line_num} in {label_path} - wrong number of values")
#                     invalid_lines += 1
#                     continue

#                 class_id = values[0]
#                 coords = [float(x) for x in values[1:]]
                
#                 # Check if coordinates are wildly out of bounds (possible parsing errors)
#                 if any(x > 2.0 or x < -1.0 for x in coords):
#                     print(f"Warning: Line {line_num} in {label_path} has severely invalid coordinates")
#                     invalid_lines += 1
#                     continue
                
#                 # Fix coordinates that are slightly out of bounds
#                 fixed_coords = [min(max(x, 0.0), 1.0) for x in coords]
                
#                 if coords != fixed_coords:
#                     modified = True
                
#                 fixed_line = f"{class_id} {' '.join(f'{x:.6f}' for x in fixed_coords)}\n"
                
#                 # Check for duplicate labels
#                 if fixed_line in seen_labels:
#                     print(f"Warning: Duplicate label found in {label_path} at line {line_num}")
#                     invalid_lines += 1
#                     continue
                
#                 seen_labels.add(fixed_line)
#                 fixed_lines.append(fixed_line)
                
#             except ValueError as e:
#                 print(f"Error parsing line {line_num} in {label_path}: {e}")
#                 invalid_lines += 1
#                 continue
        
#         if modified or invalid_lines > 0:
#             with open(label_path, 'w') as f:
#                 f.writelines(fixed_lines)
#             print(f"Fixed {label_path}: modified={modified}, invalid_lines={invalid_lines}")
#             return True
            
#     except Exception as e:
#         print(f"Error processing {label_path}: {str(e)}")
    
#     return False

# def validate_dataset_labels(label_dirs):
#     """
#     Validate all labels in the dataset.
#     """
#     print("Validating and fixing label coordinates...")
#     total_fixed = 0
#     for label_dir in label_dirs:
#         for label_file in os.listdir(label_dir):
#             if label_file.endswith('.txt'):
#                 label_path = os.path.join(label_dir, label_file)
#                 if validate_and_fix_labels(label_path):
#                     total_fixed += 1
#     print(f"Fixed coordinates in {total_fixed} files")

# def prepare_dataset(json_path, train_percent=0.46, test_percent=0.138):
#     """
#     Prepare dataset by splitting into train/test sets if not already prepared.
#     Returns the number of training and testing images.
#     """
#     test_dir = 'dataset/images/test'
#     train_dir = 'dataset/images/train'
#     label_test_dir = 'dataset/labels/test'
#     label_train_dir = 'dataset/labels/train'

#     # Check if dataset is already prepared
#     if os.path.exists(test_dir) and os.listdir(test_dir):
#         if os.path.exists(train_dir) and os.listdir(train_dir):
#             if os.path.exists(label_test_dir) and os.listdir(label_test_dir):
#                 if os.path.exists(label_train_dir) and os.listdir(label_train_dir):
#                     print("Dataset already prepared. Skipping dataset preparation.")
#                     train_count = len(os.listdir(train_dir))
#                     test_count = len(os.listdir(test_dir))
#                     return train_count, test_count

#     # If not prepared, proceed with dataset preparation
#     with open(json_path, 'r') as f:
#         data = json.load(f)

#     # Get all image IDs
#     all_images = list(data['imgs'].keys())

#     # Calculate actual numbers
#     total_images = len(all_images)
#     train_size = int(total_images * train_percent)
#     test_size = int(total_images * test_percent)

#     # Ensure enough images for training and testing
#     if train_size + test_size > total_images:
#         raise ValueError("Train and test sizes exceed total number of images.")

#     # Random sampling
#     selected_images = random.sample(all_images, train_size + test_size)
#     train_images = selected_images[:train_size]
#     test_images = selected_images[train_size:]

#     print(f"Selected {train_size} images for training and {test_size} images for testing.")

#     # Create directories if they don't exist
#     os.makedirs(train_dir, exist_ok=True)
#     os.makedirs(test_dir, exist_ok=True)
#     os.makedirs(label_train_dir, exist_ok=True)
#     os.makedirs(label_test_dir, exist_ok=True)
    
#     # Process annotations
#     for img_id in tqdm(selected_images, desc="Processing annotations"):
#         img_data = data['imgs'][img_id]
#         img_width = img_data['width']
#         img_height = img_data['height']
#         file_name = img_data['file_name']  # e.g., 'train/a4ea732cd3d5948a.jpg'

#         # Source image path
#         src_img_path = os.path.join('dataset', file_name)

#         # Destination image path
#         if img_id in train_images:
#             dest_img_path = os.path.join(train_dir, f'{img_id}.jpg')
#             subset = 'train'
#         else:
#             dest_img_path = os.path.join(test_dir, f'{img_id}.jpg')
#             subset = 'test'

#         # Copy or link the image
#         if not os.path.exists(dest_img_path):
#             try:
#                 os.link(src_img_path, dest_img_path)
#             except:
#                 # If hard link fails, use copy
#                 shutil.copy(src_img_path, dest_img_path)

#         # Get annotations for this image
#         ann_ids = data['imgToAnns'].get(img_id, [])

#         # Create YOLO format labels
#         labels = []
#         for ann_id in ann_ids:
#             ann = data['anns'][ann_id]
#             bbox = ann['bbox']

#             # Convert to YOLO format (normalized center x, center y, width, height)
#             x1, y1, x2, y2 = bbox
#             x_center = ((x1 + x2) / 2) / img_width
#             y_center = ((y1 + y2) / 2) / img_height
#             width = abs(x2 - x1) / img_width
#             height = abs(y2 - y1) / img_height

#             labels.append(f"0 {x_center} {y_center} {width} {height}")

#         # Save labels
#         label_path = os.path.join(f'dataset/labels/{subset}', f'{img_id}.txt')
#         with open(label_path, 'w') as f:
#             f.write('\n'.join(labels))
    
#     validate_dataset_labels([label_train_dir, label_test_dir])

#     print("Dataset preparation completed.")
#     train_count = len(train_images)
#     test_count = len(test_images)
#     return train_count, test_count

def save_analytics(results, train_count, test_count):
    """Save training and validation analytics to a text file."""
    analytics_path = 'runs/train/text_detection/analytics.txt'
    os.makedirs(os.path.dirname(analytics_path), exist_ok=True)
    
    with open(analytics_path, 'w') as f:
        # Dataset statistics
        f.write("Dataset Statistics:\n")
        f.write(f"- Training images: {train_count}\n")
        f.write(f"- Testing images: {test_count}\n\n")
        
        # Model performance metrics
        f.write("Model Performance Metrics:\n")
        metrics = results.mean_results()
        f.write(f"- Precision: {metrics[0]:.4f}\n")
        f.write(f"- Recall: {metrics[1]:.4f}\n")
        f.write(f"- mAP50: {metrics[2]:.4f}\n")
        f.write(f"- mAP50-95: {metrics[3]:.4f}\n")
        
        # Speed metrics
        f.write("\nSpeed Metrics:\n")
        f.write(f"- Preprocess time: {results.speed['preprocess']:.1f}ms\n")
        f.write(f"- Inference time: {results.speed['inference']:.1f}ms\n")
        f.write(f"- Postprocess time: {results.speed['postprocess']:.1f}ms\n")


# TRAINING_PARAMS = {
#     'epochs': 2,
#     'batch_size': 6,
#     'imgsz': 480,
#     'patience': 6,
#     'lr': 0.002 
# }

# def train_model(params):
#     """Train YOLOv8 model with improved learning and checkpointing."""
#     model_path = 'Yolov8/yolov8m.pt'
#     checkpoint_path = 'runs/train/text_detection/weights/last.pt'

#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model file not found at '{model_path}'. Ensure the file exists and the path is correct.")

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"Using device: {device}")

#     # Load model
#     if os.path.exists(checkpoint_path):
#         print(f"Resuming from checkpoint: {checkpoint_path}")
#         model = YOLO(checkpoint_path)
#     else:
#         print("No checkpoint found, starting fresh from the model.")
#         model = YOLO(model_path)

#     # Training
#     results = model.train(
#         data='data.yaml',
#         epochs=params['epochs'],
#         imgsz=params['imgsz'],
#         batch=params['batch_size'],
#         device=device,
#         name='text_detection',
#         save=True,
#         project='runs/train',
#         exist_ok=True,
#         workers=8,
#         half=True,
#         patience=params['patience'],
#         augment=True,
#         #resume=True
#     )

#     print("Training complete.")
#     return results












TRAINING_PARAMS = {
    'epochs': 80,
    'batch_size': 8,
    'imgsz': 640,
    'patience': 10
}

DATASET_PATH = './dataset/data/'
TRAIN_SPLIT = 11998
TEST_SPLIT = 5142
OUTPUT_PATH = './dataset/splits/'

def prepare_dataset(dataset_path, output_path, train_split, test_split):
    """Prepare dataset for YOLOv8 by creating train-test splits. Skip if splits exist."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    # Check if the splits already exist
    splits_exist = (
        os.path.exists(os.path.join(output_path, 'train/images')) and
        os.path.exists(os.path.join(output_path, 'train/labels')) and
        os.path.exists(os.path.join(output_path, 'test/images')) and
        os.path.exists(os.path.join(output_path, 'test/labels'))
    )

    if splits_exist:
        print("Dataset splits already exist. Skipping dataset preparation.")
        return train_split, test_split

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Create folders for train/test splits
    train_img_dir = os.path.join(output_path, 'train/images')
    train_label_dir = os.path.join(output_path, 'train/labels')
    test_img_dir = os.path.join(output_path, 'test/images')
    test_label_dir = os.path.join(output_path, 'test/labels')

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)

    # List all image files
    files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]
    files.sort(key=lambda x: int(x.split('.')[0]))  # Ensure proper order

    # Shuffle and split dataset
    random.seed(42)  # For reproducibility
    random.shuffle(files)
    train_files = files[:train_split]
    test_files = files[train_split:train_split + test_split]

    # Copy files to train/test directories
    for split, split_files, img_dir, label_dir in [
        ('train', train_files, train_img_dir, train_label_dir),
        ('test', test_files, test_img_dir, test_label_dir),
    ]:
        for file in split_files:
            img_path = os.path.join(dataset_path, file)
            label_path = os.path.join(dataset_path, file.replace('.jpg', '.txt'))

            shutil.copy(img_path, os.path.join(img_dir, file))
            shutil.copy(label_path, os.path.join(label_dir, file.replace('.jpg', '.txt')))

        print(f"{split.capitalize()} split: {len(split_files)} images prepared.")

    return train_split, test_split

def train_model(params):
    """Train YOLOv8 model with improved learning and checkpointing."""
    model_path = 'Yolov8/yolov8m.pt'
    checkpoint_path = 'runs/train/text_detection/weights/last.pt'

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'. Ensure the file exists and the path is correct.")

    device = 'cuda'
    print(f"Using device: {device}")
    
    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # Load model
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        model = YOLO(checkpoint_path)
    else:
        print("No checkpoint found, starting fresh from the model.")
        model = YOLO(model_path)

    # Training
    results = model.train(
        data='data.yaml',
        epochs=params['epochs'],
        imgsz=params['imgsz'],
        batch=params['batch_size'],
        device=device,
        name='text_detection',
        save=True,
        project='runs/train',
        exist_ok=True,
        workers=8,
        half=True,
        patience=params['patience'],
        #augment=True,
        amp=True,
        close_mosaic=10,
        resume=True
    )

    print("Training complete.")
    return results


def main():
    try:
        # Clear CUDA cache if needed
        clear_cuda_cache()

        # Check if PyTorch can utilize the GPU
        check_pytorch_gpu()

        print("Preparing dataset...")
        train_count, test_count = prepare_dataset(DATASET_PATH, OUTPUT_PATH, TRAIN_SPLIT, TEST_SPLIT)

        print("Training model...")
        results = train_model(TRAINING_PARAMS)

        print("Saving analytics...")
        save_analytics(results, train_count, test_count)

        print("Training completed successfully!")

    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
