from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import cv2
import numpy as np
from PIL import Image
import json
from ultralytics import YOLO
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
# MODEL_PATH = 'models/best.pt'  # Path to the trained model
MODEL_PATH = 'F:\\DL\\document_layout_analysis\\runs\\doclaynet_v8n_fast\\weights\\best.pt'
# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load the trained model (placeholder - will be loaded when model is available)
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print("Model loaded successfully!")
    else:
        print(f"Model not found at {MODEL_PATH}. Using pretrained YOLOv8n for demo.")
        model = YOLO('yolov8n.pt')  # Fallback to pretrained model
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# DocLayNet class names (11 classes)
CLASS_NAMES = [
    'Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer',
    'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'
]

# Color mapping for visualization
COLORS = [
    (255, 0, 0),    # Caption - Red
    (0, 255, 0),    # Footnote - Green
    (0, 0, 255),    # Formula - Blue
    (255, 255, 0),  # List-item - Yellow
    (255, 0, 255),  # Page-footer - Magenta
    (0, 255, 255),  # Page-header - Cyan
    (128, 0, 128),  # Picture - Purple
    (255, 165, 0),  # Section-header - Orange
    (128, 128, 128), # Table - Gray
    (0, 128, 0),    # Text - Dark Green
    (128, 0, 0)     # Title - Dark Red
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Read and process the image
        image_bytes = file.read()
        image = Image.open(BytesIO(image_bytes))
        
        # Convert PIL image to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Run inference
        results = model(opencv_image)
        
        # Process results
        predictions = []
        annotated_image = opencv_image.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Only include predictions with confidence > 0.25
                    if confidence > 0.25:
                        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class_{class_id}"
                        color = COLORS[class_id] if class_id < len(COLORS) else (255, 255, 255)
                        
                        predictions.append({
                            'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            'category': class_name,
                            'score': float(confidence)
                        })
                        
                        # Draw bounding box on image
                        cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # Add label
                        label = f"{class_name}: {confidence:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(annotated_image, (int(x1), int(y1) - label_size[1] - 10),
                                    (int(x1) + label_size[0], int(y1)), color, -1)
                        cv2.putText(annotated_image, label, (int(x1), int(y1) - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Convert annotated image to base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        annotated_image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'predictions': predictions,
            'annotated_image': f"data:image/jpeg;base64,{annotated_image_b64}",
            'total_detections': len(predictions)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH
    })

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

