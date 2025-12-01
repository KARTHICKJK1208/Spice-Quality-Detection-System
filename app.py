# Load models dynamically
import os
import cv2
import numpy as np
from flask import Flask, request, render_template, send_file, redirect, url_for
from ultralytics import YOLO
import io
from PIL import Image
import base64

app = Flask(__name__)

# Define paths for trained models (adjust paths as needed)
MODEL_PATHS = {
    "mixed_spices": {
        "yolov11": "models/best.pt",
        "frcnn": "models/best.pt"
    }
}


# Load models dynamically
models = {}
for spice, paths in MODEL_PATHS.items():
    print(f"Loading models for {spice}")
    models[f"{spice}_yolov11"] = YOLO(paths["yolov11"])
    models[f"{spice}_frcnn"] = YOLO(paths["frcnn"])
    # Print class names for debugging
    print(f"{spice}_yolov11 class names: {models[f'{spice}_yolov11'].names}")
    print(f"{spice}_frcnn class names: {models[f'{spice}_frcnn'].names}")

# Function to draw translucent label
def draw_translucent_label(img, text, position, font_scale=0.4, font_thickness=1, alpha=0.7):
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    x, y = position
    img_height, img_width = img.shape[:2]

    if y - text_height - 5 < 0:
        y = y + text_height + 5
    if x + text_width + 5 > img_width:
        x = img_width - text_width - 5

    rect_x1 = max(0, x)
    rect_y1 = max(0, y - text_height - 5)
    rect_x2 = min(img_width, x + text_width + 5)
    rect_y2 = min(img_height, y + 5)

    overlay = img.copy()
    cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255, 180), font_thickness)

def process_image(image_path, spice, model_type="yolov11"):
    # Read the image
    img = cv2.imread(image_path)
    
    # Resize image to a smaller size (e.g., 400x300)
    img = cv2.resize(img, (400, 300))
    
    # Select the appropriate model
    model = models[f"{spice}_{model_type}"]
    
    # Run YOLO detection
    print(f"Running {model_type} prediction for {spice}")
    results = model.predict(source=image_path, save=False)
    print(f"Results from {model_type}: {results}")
    
    total_pieces = 0
    good_pieces = 0
    bad_pieces = 0
    
    # Process results
    for result in results:
        boxes = result.boxes.xyxy  # Bounding box coordinates
        confidences = result.boxes.conf  # Confidence scores
        class_ids = result.boxes.cls  # Class IDs
        names = result.names  # Class names
        
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            # Adjust box coordinates based on resized image
            scale_x = img.shape[1] / cv2.imread(image_path).shape[1]
            scale_y = img.shape[0] / cv2.imread(image_path).shape[0]
            x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
            
            # Debug: Print detected class and confidence
            print(f"{model_type} detected: {names[int(cls_id)]} (class ID: {int(cls_id)}, confidence: {conf:.2f})")
            
            # Determine if the piece is "bad" based on class ID
            # Assuming class ID 1 is "good" and class ID 0 is "bad" (reversed from previous logic)
            is_bad_piece = int(cls_id) == 0  # Adjust based on your model's class IDs
            box_color = (0, 0, 255) if is_bad_piece else (0, 255, 0)  # Red for bad, Green for good
            
            # Update counts
            total_pieces += 1
            if is_bad_piece:
                bad_pieces += 1
            else:
                good_pieces += 1
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
            
            # Draw label only for mixed_spices
            if spice == "mixed_spices":
                label = f"{names[int(cls_id)]} {conf:.2f}"
                draw_translucent_label(img, label, (x1, y1))
    
    # Convert to RGB for PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"{model_type} image shape before saving: {img_rgb.shape}")
    return img_rgb, total_pieces, good_pieces, bad_pieces

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/spice/<spice>')
def spice_upload(spice):
    return render_template('upload.html', spice=spice)

@app.route('/process/<spice>', methods=['POST'])
def process_spice(spice):
    if 'files' not in request.files:
        return "No file part", 400
    
    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        return "No selected files", 400
    
    processed_images = []
    temp_paths = []
    
    # Process each uploaded file
    for file in files:
        if file.filename != '':
            temp_path = f"temp_{spice}_{len(temp_paths)}.jpg"
            file.save(temp_path)
            temp_paths.append(temp_path)
            
            # Process with both YOLOv11 and frcnn models
            img_yolov11, total_yolov11, good_yolov11, bad_yolov11 = process_image(temp_path, spice, "yolov11")
            img_frcnn, total_frcnn, good_frcnn, bad_frcnn = process_image(temp_path, spice, "frcnn")
            
            # Convert to bytes for sending
            img_io_yolov11 = io.BytesIO()
            Image.fromarray(img_yolov11).save(img_io_yolov11, 'JPEG')
            img_io_yolov11.seek(0)
            print(f"YOLOv11 image size: {len(img_io_yolov11.getvalue())} bytes")
            
            img_io_frcnn = io.BytesIO()
            Image.fromarray(img_frcnn).save(img_io_frcnn, 'JPEG')
            img_io_frcnn.seek(0)
            print(f"frcnn image size: {len(img_io_frcnn.getvalue())} bytes")
            
            # Encode images with base64
            img_base64_yolov11 = base64.b64encode(img_io_yolov11.getvalue()).decode('utf-8')
            img_base64_frcnn = base64.b64encode(img_io_frcnn.getvalue()).decode('utf-8')
            
            processed_images.append({
                "yolov11": img_base64_yolov11,
                "frcnn": img_base64_frcnn,
                "filename": file.filename,
                "total_yolov11": total_yolov11,
                "good_yolov11": good_yolov11,
                "bad_yolov11": bad_yolov11,
                "total_frcnn": total_frcnn,
                "good_frcnn": good_frcnn,
                "bad_frcnn": bad_frcnn,
                "spice": spice  # Pass the spice name for dynamic labeling
            })
    
    # Clean up temporary files
    for temp_path in temp_paths:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    return render_template('results.html', images=processed_images, spice=spice)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

