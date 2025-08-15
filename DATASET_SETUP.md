# DocLayNet Dataset Setup Guide

This guide provides detailed instructions for downloading and setting up the DocLayNet dataset for training the document layout analysis model.

## 📊 About DocLayNet Dataset

DocLayNet is a large-scale dataset for document layout analysis containing:
- **80,863 manually annotated pages**
- **11 layout classes** (Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture, Section-header, Table, Text, Title)
- **6 document categories** (financial reports, scientific articles, laws and regulations, government tenders, manuals, patents)
- **COCO format annotations** for easy integration with object detection models

## 🔗 Dataset Source

**Primary Source**: [Hugging Face - DocLayNet Dataset](https://huggingface.co/datasets/ds4sd/DocLayNet)

**Alternative Sources**:
- [IBM Research GitHub](https://github.com/DS4SD/DocLayNet)
- [Papers with Code](https://paperswithcode.com/dataset/doclaynet)

## 📥 Download Instructions

### Method 1: Direct Download from Hugging Face (Recommended)

1. **Visit the dataset page**:
   ```
   https://huggingface.co/datasets/ds4sd/DocLayNet
   ```

2. **Navigate to Files**:
   - Click on the "Files and versions" tab
   - You'll see the dataset structure with folders and files

3. **Download Required Files**:
   
   **Option A: Download Individual Components**
   - Download `COCO/` folder (contains annotations)
   - Download `PNG/` folder (contains images)
   - These are the core files needed for training

   **Option B: Use Git LFS (Large File Storage)**
   ```bash
   # Install git-lfs if not already installed
   git lfs install
   
   # Clone the dataset repository
   git clone https://huggingface.co/datasets/ds4sd/DocLayNet
   ```

### Method 2: Using Hugging Face Datasets Library

```python
from datasets import load_dataset

# Load the dataset (this will download automatically)
dataset = load_dataset("ds4sd/DocLayNet")

# Access different splits
train_data = dataset['train']
val_data = dataset['validation'] 
test_data = dataset['test']
```

### Method 3: Using wget (Command Line)

```bash
# Create data directory
mkdir -p data/DocLayNet

# Download COCO annotations
wget -O data/DocLayNet/coco_annotations.json \
  "https://huggingface.co/datasets/ds4sd/DocLayNet/resolve/main/COCO/train.json"

# Note: Image downloads require individual file URLs
# Use the Hugging Face interface or datasets library for bulk downloads
```

## 📁 Directory Structure Setup

After downloading, organize the files in the following structure:

```
document_layout_analysis/
└── data/
    └── DocLayNet/
        ├── images/
        │   ├── train/           # Training images (PNG format)
        │   ├── val/             # Validation images  
        │   └── test/            # Test images
        ├── coco_annotations.json # COCO format annotations
        └── metadata/            # Additional metadata (optional)
            ├── categories.json
            └── splits.json
```

## 🔧 Data Preparation Steps

### Step 1: Extract Downloaded Files

If you downloaded compressed files:

```bash
# Navigate to your project directory
cd document_layout_analysis

# Create data directory
mkdir -p data/DocLayNet

# Extract files (adjust paths based on your downloads)
unzip DocLayNet_images.zip -d data/DocLayNet/
unzip DocLayNet_annotations.zip -d data/DocLayNet/
```

### Step 2: Organize Image Files

The images should be organized by split:

```bash
# If images are in a single folder, organize them by split
cd data/DocLayNet

# Create split directories
mkdir -p images/{train,val,test}

# Move images to appropriate directories based on the split information
# This step may require parsing the COCO annotations to determine splits
```

### Step 3: Verify Dataset Integrity

Run the verification script:

```python
import os
import json

def verify_dataset():
    base_path = "data/DocLayNet"
    
    # Check if directories exist
    required_dirs = [
        "images/train",
        "images/val", 
        "images/test"
    ]
    
    for dir_path in required_dirs:
        full_path = os.path.join(base_path, dir_path)
        if os.path.exists(full_path):
            count = len(os.listdir(full_path))
            print(f"✓ {dir_path}: {count} images")
        else:
            print(f"✗ {dir_path}: Directory not found")
    
    # Check annotations file
    annotations_path = os.path.join(base_path, "coco_annotations.json")
    if os.path.exists(annotations_path):
        with open(annotations_path, 'r') as f:
            data = json.load(f)
            print(f"✓ Annotations: {len(data.get('images', []))} images, {len(data.get('annotations', []))} annotations")
    else:
        print("✗ coco_annotations.json: File not found")

if __name__ == "__main__":
    verify_dataset()
```

### Step 4: Run Data Preparation Script

```bash
cd utils
python data_preparation.py
```

This script will:
- Convert COCO annotations to YOLO format
- Create train/val/test splits
- Generate the `doclaynet.yaml` configuration file
- Create a subset for quick training (configurable)

## 📊 Dataset Statistics

After successful setup, you should have:

| Split | Images | Annotations | Size (approx.) |
|-------|--------|-------------|----------------|
| Train | ~65,000 | ~650,000 | ~15 GB |
| Val   | ~8,000  | ~80,000  | ~2 GB |
| Test  | ~8,000  | ~80,000  | ~2 GB |
| **Total** | **~80,863** | **~810,000** | **~20 GB** |

## 🎯 Class Distribution

The 11 layout classes and their typical distribution:

| Class | Description | Frequency |
|-------|-------------|-----------|
| Text | Regular paragraph text | ~45% |
| Title | Document titles | ~8% |
| Section-header | Section headings | ~12% |
| Table | Tabular data | ~10% |
| Picture | Images, figures | ~8% |
| List-item | List elements | ~7% |
| Caption | Image/table captions | ~4% |
| Page-header | Header content | ~2% |
| Page-footer | Footer content | ~2% |
| Formula | Mathematical formulas | ~1% |
| Footnote | Footnote text | ~1% |

## 🚨 Troubleshooting

### Common Issues and Solutions

1. **Download Timeout/Interruption**:
   ```bash
   # Resume interrupted downloads using wget
   wget -c -O filename.zip "download_url"
   ```

2. **Insufficient Disk Space**:
   - The full dataset requires ~20GB of free space
   - Consider downloading only a subset for initial testing
   - Use the data preparation script to create smaller training sets

3. **Slow Download Speeds**:
   - Use a download manager for large files
   - Consider downloading during off-peak hours
   - Try alternative mirrors if available

4. **File Corruption**:
   ```bash
   # Verify file integrity using checksums (if provided)
   md5sum downloaded_file.zip
   sha256sum downloaded_file.zip
   ```

5. **Permission Issues**:
   ```bash
   # Fix file permissions
   chmod -R 755 data/DocLayNet/
   ```

### Dataset Validation

Run this validation script to ensure everything is set up correctly:

```python
import os
import json
from PIL import Image

def validate_dataset():
    """Comprehensive dataset validation"""
    
    base_path = "data/DocLayNet"
    issues = []
    
    # 1. Check directory structure
    required_paths = [
        "images/train",
        "images/val", 
        "images/test",
        "coco_annotations.json"
    ]
    
    for path in required_paths:
        if not os.path.exists(os.path.join(base_path, path)):
            issues.append(f"Missing: {path}")
    
    # 2. Validate annotations
    try:
        with open(os.path.join(base_path, "coco_annotations.json"), 'r') as f:
            coco_data = json.load(f)
            
        # Check required COCO fields
        required_fields = ['images', 'annotations', 'categories']
        for field in required_fields:
            if field not in coco_data:
                issues.append(f"Missing COCO field: {field}")
        
        # Validate categories (should be 11 classes)
        if len(coco_data.get('categories', [])) != 11:
            issues.append(f"Expected 11 categories, found {len(coco_data.get('categories', []))}")
            
    except Exception as e:
        issues.append(f"Error reading annotations: {e}")
    
    # 3. Sample image validation
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(base_path, 'images', split)
        if os.path.exists(split_path):
            images = os.listdir(split_path)[:5]  # Check first 5 images
            for img_name in images:
                try:
                    img_path = os.path.join(split_path, img_name)
                    with Image.open(img_path) as img:
                        if img.size[0] == 0 or img.size[1] == 0:
                            issues.append(f"Invalid image size: {img_name}")
                except Exception as e:
                    issues.append(f"Cannot open image {img_name}: {e}")
    
    # Report results
    if not issues:
        print("✅ Dataset validation passed!")
        print("Your DocLayNet dataset is ready for training.")
    else:
        print("❌ Dataset validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        
    return len(issues) == 0

if __name__ == "__main__":
    validate_dataset()
```

## 📚 Additional Resources

- **Dataset Paper**: [DocLayNet: A Large Human-Annotated Dataset for Document-Layout Analysis](https://arxiv.org/abs/2206.01062)
- **Dataset Card**: [Hugging Face Dataset Card](https://huggingface.co/datasets/ds4sd/DocLayNet)
- **COCO Format**: [COCO Dataset Format](https://cocodataset.org/#format-data)
- **IBM Research Blog**: [DocLayNet Blog Post](https://research.ibm.com/blog/doclaynet-dataset)

## 💡 Tips for Efficient Usage

1. **Start Small**: Use the subset creation feature in `data_preparation.py` for initial experiments
2. **Monitor Storage**: The full dataset is large; monitor disk usage during download
3. **Backup**: Consider backing up the processed dataset after preparation
4. **Version Control**: Keep track of which version of the dataset you're using
5. **Documentation**: Document any modifications you make to the dataset structure

---

**Need Help?** Check the main README.md or open an issue in the project repository.

