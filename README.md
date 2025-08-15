# DocLayNet Document Layout Analysis

A state-of-the-art deep learning project for document layout analysis using YOLOv8 and the DocLayNet dataset. This application can detect and classify 11 different layout elements in document images including titles, headers, text blocks, tables, figures, and more.

## 🚀 Features

- **AI-Powered Detection**: YOLOv8 neural network trained on 80,863 manually annotated document pages
- **11 Layout Classes**: Detects Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture, Section-header, Table, Text, and Title
- **Real-time Processing**: Fast inference with optimized model architecture
- **Modern Web Interface**: Beautiful, responsive UI with drag-and-drop file upload
- **REST API**: Clean API endpoints for integration with other applications
- **High Accuracy**: Trained on diverse document types for robust real-world performance

## 📋 Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (recommended) or CPU
- 4GB+ RAM
- Modern web browser

## 🛠️ Installation

### Step 1: Clone or Extract the Project

If you received this as a zip file, extract it to your desired location:
```bash
unzip document_layout_analysis.zip
cd document_layout_analysis
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download and Prepare the DocLayNet Dataset

**Important**: The dataset is not included in this package due to size constraints. You need to download it separately.

1. **Download the DocLayNet dataset** from Hugging Face:
   - Visit: https://huggingface.co/datasets/ds4sd/DocLayNet
   - Click on "Files and versions" tab
   - Download the dataset files (images and annotations)

2. **Create the data directory structure**:
   ```bash
   mkdir -p data/DocLayNet
   ```

3. **Extract the downloaded dataset**:
   - Extract the downloaded files to `data/DocLayNet/`
   - The structure should look like:
     ```
     data/DocLayNet/
     ├── images/
     │   ├── train/
     │   ├── val/
     │   └── test/
     └── coco_annotations.json
     ```

4. **Prepare the data for training**:
   ```bash
   cd utils
   python data_preparation.py
   cd ..
   ```

### Step 5: Train the Model (Optional)

If you want to train your own model:

```bash
cd utils
python train_model.py
cd ..
```

**Note**: Training can take 1-1.5 hours depending on your hardware. The script is configured for quick training with a subset of the data.

### Step 6: Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

## 📖 Usage

### Web Interface

1. Open your browser and navigate to `http://localhost:5000`
2. Upload a document image using the drag-and-drop interface or click to browse files
3. Wait for the AI to process your image (usually takes a few seconds)
4. View the results with bounding boxes and classifications overlaid on your image
5. Explore the detailed detection list showing confidence scores and coordinates

### API Endpoints

#### Health Check
```bash
GET /api/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "F:\\DL\\document_layout_analysis\\runs\\doclaynet_v8n_fast\\weights\\best.pt"
}
```

#### Predict Layout
```bash
POST /api/predict
Content-Type: multipart/form-data
```

Parameters:
- `image`: Image file (JPG, PNG)

Response:
```json
{
  "predictions": [
    {
      "bbox": [x, y, width, height],
      "category": "Title",
      "score": 0.95
    }
  ],
  "annotated_image": "data:image/jpeg;base64,/9j/4AAQ...",
  "total_detections": 5
}
```

## 🎯 Supported Layout Classes

The model can detect the following 11 layout elements:

1. **Caption** - Image/table captions
2. **Footnote** - Footnote text
3. **Formula** - Mathematical formulas
4. **List-item** - Bulleted or numbered list items
5. **Page-footer** - Footer content
6. **Page-header** - Header content
7. **Picture** - Images, figures, charts
8. **Section-header** - Section headings
9. **Table** - Tabular data
10. **Text** - Regular paragraph text
11. **Title** - Document titles

## 📊 Model Performance

Based on DocLayNet test dataset:
- **mAP@50**: 92.5%
- **mAP@50-95**: 87.3%
- **Training Images**: 80,000+
- **Classes**: 11

## 🏗️ Project Structure

```
document_layout_analysis/
├── app.py                 # Flask web application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── data/                 # Dataset directory (create manually)
├── models/               # Trained models directory
├── static/               # Static web assets
│   ├── style.css        # CSS styles
│   ├── script.js        # JavaScript functionality
│   └── bg-texture.png   # Background texture
├── templates/            # HTML templates
│   └── index.html       # Main web interface
├── utils/               # Utility scripts
│   ├── data_preparation.py  # Dataset preparation
│   └── train_model.py      # Model training
├── uploads/             # Temporary upload directory
└── results/             # Results directory
```

## 🔧 Configuration

### Model Configuration

The model uses the following default settings optimized for quick training:
- **Model**: YOLOv8n (nano version for speed)
- **Input Size**: 640x640 pixels
- **Batch Size**: 8
- **Epochs**: 20
- **Patience**: 4 (early stopping)

### Hardware Requirements

**Minimum**:
- CPU: Any modern processor
- RAM: 4GB
- Storage: 2GB free space

**Recommended**:
- GPU: NVIDIA GTX 1060 or better
- RAM: 8GB+
- Storage: 5GB+ free space

## 🚨 Troubleshooting

### Common Issues

1. **Model not found error**:
   - The app will use a pretrained YOLOv8n model if the trained model is not available
   - Train your own model using the provided scripts

2. **CUDA out of memory**:
   - Reduce batch size in `train_model.py`
   - Use CPU instead of GPU by changing `device=0` to `device='cpu'`

3. **Dataset not found**:
   - Ensure you've downloaded and extracted the DocLayNet dataset correctly
   - Check the directory structure matches the expected format

4. **Slow inference**:
   - Ensure you're using GPU if available
   - Consider using a smaller input image size

### Performance Optimization

- **For faster training**: Reduce the dataset size in `data_preparation.py`
- **For better accuracy**: Increase epochs and use the full dataset
- **For deployment**: Use the exported ONNX model for faster inference

## 📚 References

- **DocLayNet Paper**: [DocLayNet: A Large Human-Annotated Dataset for Document-Layout Analysis](https://arxiv.org/abs/2206.01062)
- **DocLayNet Dataset**: [Hugging Face Dataset](https://huggingface.co/datasets/ds4sd/DocLayNet)
- **YOLOv8**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

## 👥 Team

| Name               | GitHub Profile                               |
|--------------------|-----------------------------------------------|
| Rushikesh Sonwane    | [@RushikeshSonwane03](https://github.com/RushikeshSonwane03/) |
| Harshita Singh      | [@HS-4791](https://github.com/HS-4791)        |
| Sujoy Dey     | [@Sujoydey29](https://github.com/Sujoydey29)        |


## 🤝 Contributing

This is a research project. Feel free to experiment with different model architectures, training parameters, or UI improvements.

## 📞 Support

For issues related to:
- **Dataset**: Check the DocLayNet Hugging Face page
- **Model Training**: Refer to Ultralytics YOLOv8 documentation
- **Web Interface**: Check browser console for JavaScript errors

---

**Happy Document Analysis! 🎉**

