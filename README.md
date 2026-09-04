# Maryon : Marine Pollution Detection Using Sentinel-2 Satellite Imagery

## Overview

AquaSense is a deep learning-based semantic segmentation project designed to detect marine pollution from Sentinel-2 satellite imagery. The project uses the MADOS dataset and a U-Net architecture to generate pixel-wise pollution maps.

This repository currently contains the implementation of the **Sentinel-2 (Optical) Module**. Future work will integrate a Sentinel-1 (SAR) module and a multimodal fusion architecture.

---

## Features

- MADOS dataset loader
- Sentinel-2 optical image processing
- PyTorch Dataset & DataLoader
- U-Net semantic segmentation model
- Model checkpointing
- Best model saving
- Training pipeline
- Prediction and visualization
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Mean IoU

---

## Project Architecture

```
Sentinel-2 Images
        │
        ▼
Image Preprocessing
        │
        ▼
PyTorch Dataset
        │
        ▼
U-Net Encoder
        │
        ▼
U-Net Decoder
        │
        ▼
Softmax Layer
        │
        ▼
Marine Pollution Segmentation Map
```

---

## Dataset

**Dataset:** MADOS (Marine Debris Detection Dataset)

Current implementation uses:

- RGB Images
- Segmentation Masks

Future implementation will support multispectral Sentinel-2 bands:

- B2 (Blue)
- B3 (Green)
- B4 (Red)
- B8 (NIR)
- B11 (SWIR1)
- B12 (SWIR2)

---

## Current Workflow

```
Download Dataset
        │
        ▼
Extract Dataset
        │
        ▼
Load Images & Masks
        │
        ▼
Resize Images (256×256)
        │
        ▼
Normalize Images
        │
        ▼
Create Dataset & DataLoader
        │
        ▼
Train U-Net
        │
        ▼
Save Checkpoints
        │
        ▼
Generate Predictions
        │
        ▼
Evaluate Model
```

---

## Technologies Used

- Python
- PyTorch
- OpenCV
- Rasterio
- NumPy
- Albumentations
- Scikit-Image
- Matplotlib
- Google Colab

---

## Project Structure

```
AquaSense/
│
├── dataset/
├── checkpoints/
├── predictions/
├── notebooks/
├── README.md
│
├── dataset_loader.py
├── train.py
├── predict.py
├── metrics.py
├── model.py
└── requirements.txt
```

---

## Evaluation Metrics

The model is evaluated using:

- Pixel Accuracy
- Precision
- Recall
- F1 Score
- Mean Intersection over Union (mIoU)

---

## Current Results

| Metric | Value |
|---------|-------|
| Accuracy | ~98% |
| IoU | ~0.22 |
| Precision | ~0.23 |
| Recall | ~0.24 |
| F1 Score | ~0.24 |

> **Note:** High pixel accuracy is mainly due to class imbalance. IoU and F1 Score provide a more reliable measure of segmentation performance.

---

## Future Work

The next development stages include:

### Sentinel-2 Module

- Read all Sentinel-2 bands (B2, B3, B4, B8, B11, B12)
- Resample 20 m bands to 10 m
- Build multispectral tensors
- Histogram Equalization
- Bilateral Filtering
- Data Augmentation

### Sentinel-1 Module

- Read VV and VH SAR bands
- SAR preprocessing
- Speckle noise reduction
- SAR feature extraction

### Fusion Module

- Dual Encoder Architecture
- Cross-Attention Fusion
- Feature Pyramid Network (FPN)
- Enhanced U-Net Decoder
- Improved marine pollution segmentation

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/AquaSense.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run Training

```bash
python train.py
```

---

## Run Prediction

```bash
python predict.py
```

---

## Author

**Manoj I**

Bachelor of Engineering – Computer Science

Marine Pollution Detection using Deep Learning and Remote Sensing

---

## License

This project is intended for academic and research purposes.
