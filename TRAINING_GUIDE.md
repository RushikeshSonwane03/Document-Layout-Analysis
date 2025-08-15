# Model Training Guide

This guide provides comprehensive instructions for training the YOLOv8 model on the DocLayNet dataset for document layout analysis.

## 🎯 Training Overview

The training process involves:
1. **Data Preparation**: Converting DocLayNet COCO format to YOLO format
2. **Model Configuration**: Setting up YOLOv8 with optimal parameters
3. **Training Execution**: Running the training with monitoring
4. **Evaluation**: Assessing model performance
5. **Model Export**: Preparing the model for deployment

## ⚙️ Training Configuration

### Hardware Requirements

**Minimum Requirements**:
- **GPU**: NVIDIA GTX 1060 (4GB VRAM) or equivalent
- **RAM**: 8GB system RAM
- **Storage**: 5GB free space
- **CPU**: Multi-core processor (4+ cores recommended)

**Recommended Setup**:
- **GPU**: NVIDIA RTX 3060 (8GB VRAM) or better
- **RAM**: 16GB+ system RAM
- **Storage**: 10GB+ free space (SSD preferred)
- **CPU**: 8+ core processor

**Time Estimates** (for subset training):
- **RTX 4090**: 30-45 minutes
- **RTX 3080**: 45-60 minutes
- **RTX 3060**: 60-90 minutes
- **GTX 1060**: 90-120 minutes
- **CPU Only**: 4-6 hours (not recommended)

### Model Parameters

The training script uses these optimized parameters:

```python
# Training Configuration
MODEL = "yolov8n.pt"          # Nano version for speed
IMAGE_SIZE = 640              # Input image size
BATCH_SIZE = 8                # Batch size (adjust based on VRAM)
EPOCHS = 20                   # Number of training epochs
PATIENCE = 4                  # Early stopping patience
DEVICE = 0                    # GPU device (0 for first GPU, 'cpu' for CPU)
WORKERS = 2                   # Number of data loading workers
AMP = True                    # Automatic Mixed Precision
FREEZE = 10                   # Freeze first 10 layers initially
```

## 🚀 Quick Start Training

### Step 1: Prepare the Dataset

```bash
# Ensure dataset is downloaded and extracted to data/DocLayNet/
# Run the data preparation script
cd utils
python data_preparation.py
cd ..
```

This creates:
- YOLO-formatted labels in `doclaynet_yolo/labels/`
- Organized images in `doclaynet_yolo/images/`
- Configuration file `doclaynet.yaml`

### Step 2: Start Training

```bash
cd utils
python train_model.py
```

### Step 3: Monitor Training

The training script will output:
- Real-time loss values
- Validation metrics (mAP@50, mAP@50-95)
- Training progress and ETA
- Best model checkpoints

## 📊 Understanding Training Output

### Training Metrics

**Loss Values**:
- `box_loss`: Bounding box regression loss
- `cls_loss`: Classification loss  
- `dfl_loss`: Distribution focal loss
- `total_loss`: Combined loss

**Validation Metrics**:
- `mAP@50`: Mean Average Precision at IoU threshold 0.5
- `mAP@50-95`: Mean Average Precision averaged over IoU thresholds 0.5-0.95
- `Precision`: True positives / (True positives + False positives)
- `Recall`: True positives / (True positives + False negatives)

### Expected Performance

**Target Metrics** (on DocLayNet test set):
- `mAP@50`: 85-95%
- `mAP@50-95`: 75-85%
- `Precision`: 80-90%
- `Recall`: 80-90%

**Per-Class Performance** (typical ranges):
- **Text**: mAP@50 > 90% (most common class)
- **Title**: mAP@50 > 85%
- **Table**: mAP@50 > 80%
- **Picture**: mAP@50 > 85%
- **Section-header**: mAP@50 > 80%
- **List-item**: mAP@50 > 75%
- **Caption**: mAP@50 > 70%
- **Page-header/footer**: mAP@50 > 65%
- **Formula**: mAP@50 > 60% (rare class)
- **Footnote**: mAP@50 > 60% (rare class)

## 🔧 Advanced Training Options

### Custom Training Script

Create your own training script for more control:

```python
from ultralytics import YOLO
import torch

def custom_train():
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model = YOLO("yolov8n.pt")
    
    # Custom training parameters
    results = model.train(
        data="doclaynet.yaml",
        epochs=50,                    # More epochs for better accuracy
        imgsz=896,                   # Larger image size for better detection
        batch=4,                     # Smaller batch for larger images
        lr0=0.01,                    # Initial learning rate
        lrf=0.1,                     # Final learning rate factor
        momentum=0.937,              # SGD momentum
        weight_decay=0.0005,         # Weight decay
        warmup_epochs=3,             # Warmup epochs
        warmup_momentum=0.8,         # Warmup momentum
        box=7.5,                     # Box loss gain
        cls=0.5,                     # Classification loss gain
        dfl=1.5,                     # DFL loss gain
        pose=12.0,                   # Pose loss gain (not used)
        kobj=1.0,                    # Keypoint objectness loss gain
        label_smoothing=0.0,         # Label smoothing epsilon
        nbs=64,                      # Nominal batch size
        hsv_h=0.015,                 # HSV-Hue augmentation
        hsv_s=0.7,                   # HSV-Saturation augmentation
        hsv_v=0.4,                   # HSV-Value augmentation
        degrees=0.0,                 # Rotation degrees (0.0-180.0)
        translate=0.1,               # Translation fraction (0.0-1.0)
        scale=0.5,                   # Scaling factor (0.0-1.0)
        shear=0.0,                   # Shear degrees (0.0-10.0)
        perspective=0.0,             # Perspective factor (0.0-0.001)
        flipud=0.0,                  # Vertical flip probability
        fliplr=0.5,                  # Horizontal flip probability
        mosaic=1.0,                  # Mosaic probability
        mixup=0.0,                   # Mixup probability
        copy_paste=0.0,              # Copy-paste probability
        device=device,
        workers=4,
        project="runs/train",
        name="doclaynet_custom",
        exist_ok=True,
        pretrained=True,
        optimizer='SGD',             # Optimizer (SGD, Adam, AdamW)
        verbose=True,
        seed=0,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=False,
        close_mosaic=10,
        resume=False,
        amp=True,
        fraction=1.0,
        profile=False,
        freeze=None,
        multi_scale=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        split='val',
        save_json=False,
        save_hybrid=False,
        conf=None,
        iou=0.7,
        max_det=300,
        half=False,
        dnn=False,
        plots=True,
        source=None,
        show=False,
        save_txt=False,
        save_conf=False,
        save_crop=False,
        show_labels=True,
        show_conf=True,
        vid_stride=1,
        stream_buffer=False,
        line_width=None,
        visualize=False,
        augment=False,
        agnostic_nms=False,
        classes=None,
        retina_masks=False,
        boxes=True,
        format='torchscript',
        keras=False,
        optimize=False,
        int8=False,
        dynamic=False,
        simplify=False,
        opset=None,
        workspace=4,
        nms=False,
        lr_scheduler='linear',
        patience=50,
        save=True,
        save_period=-1,
        cache=False,
        device_count=1,
        batch_size=None,
        imgsz_val=None,
        rect_val=False,
        save_dir='runs/train/doclaynet_custom'
    )
    
    return results

if __name__ == "__main__":
    results = custom_train()
```

### Hyperparameter Tuning

Use Ultralytics' built-in hyperparameter tuning:

```python
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

# Tune hyperparameters
model.tune(
    data="doclaynet.yaml",
    epochs=30,
    iterations=300,
    optimizer='AdamW',
    plots=False,
    save=False,
    val=False
)
```

### Multi-GPU Training

For multiple GPUs:

```python
# Set environment variable for multi-GPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  # Use GPUs 0,1,2,3

from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.train(
    data="doclaynet.yaml",
    epochs=50,
    imgsz=640,
    batch=32,  # Larger batch size for multi-GPU
    device=[0,1,2,3],  # Specify multiple GPUs
    workers=8
)
```

## 📈 Training Monitoring

### TensorBoard Integration

Monitor training with TensorBoard:

```bash
# Install tensorboard if not already installed
pip install tensorboard

# Start TensorBoard (run in separate terminal)
tensorboard --logdir runs/train

# Open browser to http://localhost:6006
```

### Weights & Biases Integration

For advanced experiment tracking:

```bash
# Install wandb
pip install wandb

# Login to wandb
wandb login

# Training with wandb logging
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='doclaynet.yaml', epochs=20, project='wandb')
"
```

### Custom Callbacks

Add custom callbacks for monitoring:

```python
from ultralytics import YOLO
from ultralytics.utils.callbacks import default_callbacks

def on_train_epoch_end(trainer):
    """Custom callback for end of training epoch"""
    print(f"Epoch {trainer.epoch}: Loss = {trainer.loss}")
    
    # Save checkpoint every 5 epochs
    if trainer.epoch % 5 == 0:
        trainer.save_model()

# Add custom callback
default_callbacks['on_train_epoch_end'].append(on_train_epoch_end)

# Train with custom callbacks
model = YOLO("yolov8n.pt")
model.train(data="doclaynet.yaml", epochs=20)
```

## 🎯 Optimization Strategies

### Memory Optimization

For limited VRAM:

```python
# Reduce batch size
batch_size = 4  # or even 2

# Use gradient accumulation
# Effective batch size = batch_size * accumulate
accumulate = 4  # Accumulate gradients over 4 batches

# Reduce image size
imgsz = 512  # Instead of 640

# Use mixed precision
amp = True

# Reduce workers
workers = 1
```

### Speed Optimization

For faster training:

```python
# Use larger batch size (if VRAM allows)
batch_size = 16

# Increase workers
workers = 8

# Use smaller model
model = YOLO("yolov8n.pt")  # Nano (fastest)

# Reduce validation frequency
val_period = 5  # Validate every 5 epochs instead of every epoch

# Use cached dataset
cache = True
```

### Accuracy Optimization

For better performance:

```python
# Use larger model
model = YOLO("yolov8s.pt")  # Small or medium

# Increase image size
imgsz = 896

# More epochs
epochs = 100

# Better augmentation
hsv_h = 0.015
hsv_s = 0.7
hsv_v = 0.4
degrees = 10.0
translate = 0.2
scale = 0.9
```

## 🔍 Troubleshooting Training Issues

### Common Problems and Solutions

1. **CUDA Out of Memory**:
   ```python
   # Reduce batch size
   batch = 4  # or 2
   
   # Reduce image size
   imgsz = 512
   
   # Use CPU if necessary
   device = 'cpu'
   ```

2. **Slow Training**:
   ```python
   # Check GPU utilization
   nvidia-smi
   
   # Increase workers (if CPU/RAM allows)
   workers = 4
   
   # Use cached dataset
   cache = True
   ```

3. **Poor Convergence**:
   ```python
   # Adjust learning rate
   lr0 = 0.001  # Lower learning rate
   
   # Increase warmup
   warmup_epochs = 5
   
   # Check data quality
   # Verify annotations are correct
   ```

4. **Overfitting**:
   ```python
   # Add regularization
   weight_decay = 0.001
   dropout = 0.1
   
   # Increase augmentation
   mixup = 0.1
   copy_paste = 0.1
   
   # Early stopping
   patience = 10
   ```

5. **Class Imbalance**:
   ```python
   # Use focal loss
   fl_gamma = 2.0
   
   # Adjust class weights in loss function
   # (requires custom implementation)
   
   # Oversample rare classes in data preparation
   ```

## 📊 Model Evaluation

### Validation During Training

The model automatically validates during training. Key metrics to watch:

- **mAP@50**: Should steadily increase
- **mAP@50-95**: More strict metric, increases slower
- **Loss**: Should steadily decrease
- **Precision/Recall**: Should improve over time

### Post-Training Evaluation

```python
from ultralytics import YOLO

# Load trained model
model = YOLO("runs/train/doclaynet_v8n_fast/weights/best.pt")

# Evaluate on test set
results = model.val(
    data="doclaynet.yaml",
    split='test',
    imgsz=640,
    batch=16,
    conf=0.25,
    iou=0.7,
    max_det=300,
    half=False,
    device=0,
    dnn=False,
    plots=True,
    save_json=True,
    save_hybrid=False
)

# Print results
print(f"mAP@50: {results.box.map50}")
print(f"mAP@50-95: {results.box.map}")
print(f"Precision: {results.box.mp}")
print(f"Recall: {results.box.mr}")
```

### Custom Evaluation Metrics

```python
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_per_class(model, test_data):
    """Evaluate model performance per class"""
    
    results = model.val(data="doclaynet.yaml", split='test')
    
    # Get per-class metrics
    class_names = ['Caption', 'Footnote', 'Formula', 'List-item', 
                   'Page-footer', 'Page-header', 'Picture', 
                   'Section-header', 'Table', 'Text', 'Title']
    
    # Print per-class AP
    for i, class_name in enumerate(class_names):
        ap50 = results.box.maps[i]  # AP@50 for class i
        print(f"{class_name}: AP@50 = {ap50:.3f}")
    
    return results

# Run evaluation
results = evaluate_per_class(model, test_data)
```

## 🚀 Model Export and Deployment

### Export Trained Model

```python
from ultralytics import YOLO

# Load trained model
model = YOLO("runs/train/doclaynet_v8n_fast/weights/best.pt")

# Export to different formats
model.export(format='onnx')        # ONNX format
model.export(format='torchscript') # TorchScript
model.export(format='tflite')      # TensorFlow Lite
model.export(format='coreml')      # Core ML (macOS)
model.export(format='engine')      # TensorRT (NVIDIA)
```

### Model Optimization

```python
# Export with optimizations
model.export(
    format='onnx',
    optimize=True,      # Optimize for inference
    half=True,          # Use FP16 precision
    dynamic=True,       # Dynamic input shapes
    simplify=True       # Simplify model
)
```

## 📚 Additional Resources

- **YOLOv8 Documentation**: [Ultralytics Docs](https://docs.ultralytics.com/)
- **Training Tips**: [YOLOv8 Training Guide](https://docs.ultralytics.com/modes/train/)
- **Hyperparameter Tuning**: [Ultralytics Tuning](https://docs.ultralytics.com/modes/tune/)
- **Model Export**: [Export Guide](https://docs.ultralytics.com/modes/export/)

---

**Happy Training! 🎉**

Remember: Good training takes time and experimentation. Start with the default settings and gradually optimize based on your specific needs and hardware constraints.

