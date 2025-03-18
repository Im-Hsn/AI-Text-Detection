# AI-Text-Detection

## Overview
AI-Text-Detection is a machine learning application that detects and extracts text from images using YOLOv8 object detection and EasyOCR for text recognition. The project includes both a training pipeline for creating custom text detection models and a web application for easy text extraction from uploaded images.

## Features
- **Text Detection**: Locates text areas in images using YOLOv8
- **Text Recognition**: Extracts text content using EasyOCR
- **Web Interface**: User-friendly interface to upload images and view results
- **PDF Report Generation**: Creates detailed reports with detected text and bounding boxes
- **GPU Acceleration**: Utilizes CUDA for improved performance when available
- **Custom Model Training**: Pipeline for training on your own datasets

## Project Structure
```
AI-Text-Detection/
├── app.py                  # Flask web application
├── Train.py                # YOLOv8 model training script
├── data.yaml               # Dataset configuration for YOLOv8
├── Dockerfile              # Docker configuration
├── Procfile                # Heroku deployment configuration
├── requirements.txt        # Project dependencies
├── .dockerignore           # Docker build exclusions
├── .gitignore              # Git exclusions
├── Prepare/
│   └── Download_Yolov8.py  # Script to download YOLOv8 model
├── templates/
│   ├── index.html          # Upload page template
│   └── results.html        # Results display template
├── Yolov8/                 # YOLOv8 model files
│   └── yolov8m.pt          # YOLOv8 medium pre-trained model
├── uploads/                # Temporary storage for uploaded images
├── static/                 # Static files for web application
├── dataset/                # Directory for training data
│   └── splits/             # Train/test data splits
└── models/                 # Trained model storage
    └── best_model.pt       # Best trained model (used for inference)
```

## Requirements
- Python 3.10
- PyTorch 2.1.2
- Ultralytics YOLOv8 8.3.40
- OpenCV 4.8.0
- Flask 3.0.0
- EasyOCR 1.7.2
- FPDF 1.7.2
- CUDA-compatible GPU (recommended for training)

## Installation

### Option 1: Local Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/AI-Text-Detection.git
cd AI-Text-Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Option 2: Docker Installation
1. Build the Docker image:
```bash
docker build -t ai-text-detection .
```

2. Run the container:
```bash
docker run -p 5000:5000 ai-text-detection
```

## Usage

### Training a Custom Model

1. Prepare your dataset in the required format:
   - Images in `dataset/splits/train/images` and `dataset/splits/test/images`
   - Labels in `dataset/splits/train/labels` and `dataset/splits/test/labels`
   - Labels should be in YOLO format (class x_center y_center width height)

2. Download the YOLOv8 pre-trained model:
```bash
python Prepare/Download_Yolov8.py
```

3. Run the training script:
```bash
python Train.py
```

4. The best model will be saved to `runs/train/text_detection/weights/best.pt`

#### Training Configuration
You can modify training parameters in `Train.py` by adjusting the `TRAINING_PARAMS` dictionary:
```python
TRAINING_PARAMS = {
    'epochs': 80,
    'batch_size': 8,
    'imgsz': 640,
    'patience': 10
}
```

### Using the Web Application

1. Start the Flask application:
```bash
python app.py
```

2. Open your browser and navigate to `http://127.0.0.1:5000`

3. Upload an image containing text

4. View the detection results with:
   - Bounding boxes highlighting detected text areas
   - Extracted text content
   - Option to download a detailed PDF report

## Model Information

The project uses YOLOv8, a state-of-the-art object detection model that offers an excellent balance between accuracy and speed. For text recognition, EasyOCR is employed to extract text from detected regions.

### Pre-trained Model
The application uses YOLOv8m (medium) as the base model, which is then fine-tuned on text detection datasets.

## Deployment

### Heroku Deployment
The included `Procfile` allows for easy deployment to Heroku:
```bash
heroku create your-app-name
git push heroku master
```

### Docker Deployment
The application can be containerized using the included Dockerfile:
```bash
docker build -t ai-text-detection .
docker run -p 5000:5000 ai-text-detection
```

## Performance Considerations
- GPU acceleration significantly improves both training and inference speed
- The web application will automatically use GPU if available
- For large datasets or real-time applications, consider using a more powerful GPU

## Future Improvements
- Multi-language text recognition support
- Improved handling of rotated text
- API endpoints for programmatic access
- Fine-tuning options through the web interface

## Acknowledgements
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [PyTorch](https://pytorch.org/)
- [Flask](https://flask.palletsprojects.com/)