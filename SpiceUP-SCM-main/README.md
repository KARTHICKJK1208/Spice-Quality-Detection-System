## Dataset Sources

1. Clove Detection
   - Custom dataset with YOLO-format annotations
   - Contains images and labels for clove quality assessment
   - Classes defined in `classes.txt`

2. Chili Detection
   - Custom dataset of chili images
   - Used for both deep learning (Faster R-CNN/YOLOv11l) and HSV-based quality detection
   - Source: Manually collected and annotated

3. Cardamom, Pepper, and Mixed Spices Detection
   - Custom datasets for each spice type
   - Trained models stored at system.
   - Annotated for quality classification (good/bad pieces)

4. General Notes
   - Datasets are proprietary and collected locally and labeled using label-studios.
   - Annotations include bounding boxes and quality labels (e.g., good/bad)


## Hardware Requirements

### Development System
- Processor: Intel Core i9 13th Gen.
- RAM: 32GB (recommended for model training and inference)
- Storage: 512GB SSD (to store datasets and trained models)
- Operating System: Windows 10/11 or Linux (e.g., Ubuntu via Colab)

### Model Training Environment
- GPU: NVIDIA GPU (e.g., Tesla T4 via Google Colab or local CUDA-enabled GPU)
- VRAM: 8GB+ (required for Faster R-CNN and YOLOv11l training)
- Alternative: CPU-only inference possible but slower


## Software Requirements

### Core Technologies
- Python: 3.11 or higher
- **Development Environment**: Google Colab, Jupyter Notebook, or local Python IDE

### Deep Learning Frameworks
- PyTorch: 1.9.0+ (for Faster R-CNN and YOLOv11l)
- Torchvision: 0.10.0+ (for Faster R-CNN utilities)
- Ultralytics YOLO: Latest version (for YOLOv11l)

### Computer Vision & Image Processing
- OpenCV: 4.5.3+ (for image processing and visualization)
- Pillow (PIL): 8.0.0+ (for image handling)
- NumPy: 1.20.3+ (for numerical operations)
- imutils: 0.5.4+ (for contour utilities in HSV detection)

### Web Application
- Flask: 2.0.0+ (for the spice detection web app)
- Base64: Standard library (for image encoding)

### Additional Libraries
- Matplotlib: 3.4.3+ (optional, for visualization if extended)
- IPython: 7.0.0+ (optional, for displaying results in notebooks)
- YAML: 5.4.1+ (for creating `data.yaml` in YOLOv11l)


## Execution Instructions

### 1. Setup Environment

#### For Deep Learning (Faster R-CNN and YOLOv11l)
- Option 1: Google Colab
  - Open Google Colab
  - Install required libraries in the first cell:
    ```bash
    !pip install torch torchvision ultralytics opencv-python pillow numpy pyyaml
    ```
  - Upload datasets and scripts (e.g., `train_val_split.py`) to Colab

- Option 2: Local Setup
  - Install Python 3.8+ and required libraries:
    ```bash
    pip install torch torchvision ultralytics opencv-python pillow numpy pyyaml
    ```

#### For HSV Detection
- Install OpenCV and dependencies:
  ```bash
  pip install opencv-python numpy imutils
  ```

#### For Web Application
- Install Flask and dependencies:
  ```bash
  pip install flask ultralytics opencv-python pillow numpy
  ```

### 2. Prepare Datasets and Models

#### Faster R-CNN
- Place dataset in YOLO format (images and labels) at system.
- Update paths in the code if different
- Pre-trained model: `fasterrcnn_resnet50_fpn` (downloaded automatically via PyTorch)

#### YOLOv11l
- Unzip dataset to `/content/custom_data/`
- Run `train_val_split.py` to split data:
  ```bash
  python train_val_split.py --datapath="/content/custom_data" --train_pct=0.9
  ```
- Generate `data.yaml` using the provided Python function

#### HSV Detection
- Ensure spices images (e.g., `IMG_20241102_122_14.jpg`) are accessible locally

#### Web Application
- Update `MODEL_PATHS` in the Flask app with correct paths to your trained `.pt` files

### 3. Train Models (Optional)

#### Faster R-CNN
- Run the training loop in a notebook or script:
  - Adjust `num_epochs`, dataset paths, and hyperparameters as needed
  - Models saved to `/system/models/` (`last.pt` and `best.pt`)

#### YOLOv11l
- Train using the YOLO CLI:
  ```bash
  yolo detect train data=/content/data.yaml model=yolo11l.pt epochs=60 imgsz=640
  ```
- Trained weights saved to `runs/detect/train/weights/`

### 4. Run Inference

#### Faster R-CNN
- Load the model and test on an image:
  - Example: `/content/e784b6fd-IMG_20250308_150356_aug9.jpg`
  - Visualize results using OpenCV

#### YOLOv11l
- Predict on validation images:
  ```bash
  yolo detect predict model=runs/detect/train/weights/best.pt source=data/validation/images save=True
  ```
- Evaluate metrics using `model.val()`

#### HSV Detection
- Run the script with an image path:
  ```bash
  python hsv_script.py
  ```

### 5. Launch Web Application
- Save `index.html`, `upload.html`, and `results.html` in a `templates/` folder
- Run the Flask app:
  ```bash
  python spice_detection_app.py
  ```
- Access at `http://localhost:5000` or the public IP if hosted
- Upload spice images and view results for both YOLOv11l and Faster R-CNN

### 6. Using the Application
- Navigate to `/spice/<spice_name>` (e.g., `/spice/clove`)
- Upload images via the interface
- View processed images with bounding boxes and quality metrics (total, good, bad pieces)
