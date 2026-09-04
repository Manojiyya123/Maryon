# Maryon: Marine Pollution Detection with Sentinel-2 and U-Net

## Overview

Maryon contains a U-Net semantic-segmentation pipeline for detecting marine debris and related ocean classes in MARIDA Sentinel-2 satellite imagery. The model produces a class prediction for every pixel in a 256 x 256 image patch.

This repository currently contains the Sentinel-2 optical U-Net module. It does not contain the MARIDA dataset.

## Model Input and Output

The model expects an 11-band GeoTIFF in this order:

```text
440, 490, 560, 665, 705, 740, 783, 842, 865, 1600, 2200 nm
```

The default configuration merges Mixed Water, Wakes, Cloud Shadows, and Waves into Marine Water. It therefore predicts 11 classes:

```text
1  Marine Debris
2  Dense Sargassum
3  Sparse Sargassum
4  Natural Organic Material
5  Ship
6  Clouds
7  Marine Water
8  Sediment-Laden Water
9  Foam
10 Turbid Water
11 Shallow Water
```

Marine Debris is class value `1` in generated prediction masks.

## Workflow

```text
MARIDA GeoTIFF patches
        |
        v
Load image and class mask
        |
        v
Replace NaN values with band means
        |
        v
Apply training augmentation and band normalization
        |
        v
Train U-Net with weighted cross-entropy
        |
        v
Save epoch checkpoints
        |
        v
Evaluate on the test split or predict one image
```

## Dataset Setup

Download MARIDA from [Zenodo](https://doi.org/10.5281/zenodo.5151941) and extract it into a local `data/` directory. The required structure is:

```text
data/
├── patches/
│   └── S2_DATE_TILE/
│       ├── S2_DATE_TILE_CROP.tif
│       ├── S2_DATE_TILE_CROP_cl.tif
│       └── S2_DATE_TILE_CROP_conf.tif
└── splits/
    ├── train_X.txt
    ├── val_X.txt
    └── test_X.txt
```

The dataset is intentionally excluded from this repository because it is large. `labels_mapping.txt` is used by the separate multi-label task and is not required by U-Net.

## Project Structure

```text
Maryon/
├── semantic_segmentation/
│   └── unet/
│       ├── dataloader.py          # Dataset loading and preprocessing
│       ├── evaluation.py          # Test evaluation and mask generation
│       ├── predict.py             # Prediction for one 11-band GeoTIFF
│       ├── train.py               # Training entry point
│       ├── unet.py                # U-Net architecture
│       └── trained_models/
│           └── 45/model.pth       # Trained checkpoint
└── utils/
    ├── assets.py                 # Class and band mappings
    └── metrics.py                # Accuracy and segmentation metrics
```

## Preprocessing

`dataloader.py` performs the preprocessing used by training and evaluation:

1. Loads the 11-band image and matching `_cl.tif` mask.
2. Merges four water-related classes into Marine Water.
3. Converts mask values from 1-based labels to zero-based PyTorch targets.
4. Replaces NaN pixels with the mean for their spectral band.
5. During training, applies random rotations of -90, 0, 90, or 180 degrees and random horizontal flips.
6. Normalizes each band using the dataset mean and standard deviation.

Validation and test data use normalization but no random augmentation.

## Installation

The original project used Python 3.7, PyTorch 1.7, and older geospatial packages. A modern Python environment with PyTorch, Torchvision, Rasterio, NumPy, Pandas, scikit-learn, scikit-image, tqdm, and TensorBoard is required.

Example installation:

```bash
python -m pip install torch torchvision tensorboard tqdm numpy pandas scikit-learn scikit-image rasterio
```

## Train the Model

Run from the repository root:

```bash
python semantic_segmentation/unet/train.py --epochs 45 --batch 2 --num_workers 1
```

On Windows PowerShell, use the selected Python executable if necessary:

```powershell
& "C:/path/to/python.exe" semantic_segmentation/unet/train.py --epochs 45 --batch 2 --num_workers 1
```

The script trains on `train_X.txt`, validates on `val_X.txt`, and saves a checkpoint after each evaluation epoch:

```text
semantic_segmentation/unet/trained_models/1/model.pth
...
semantic_segmentation/unet/trained_models/45/model.pth
```

For a quick smoke test:

```bash
python semantic_segmentation/unet/train.py --epochs 1 --batch 2 --num_workers 1
```

## Evaluate the Model

Evaluate the final checkpoint on the test split:

```bash
python semantic_segmentation/unet/evaluation.py \
  --model_path semantic_segmentation/unet/trained_models/45/model.pth \
  --batch 2
```

The evaluation prints accuracy, Macro F1, Mean IoU, Marine Debris recall, Marine Debris IoU, and a confusion matrix. It also generates georeferenced prediction masks in:

```text
data/predicted_unet/
```

The test split must be used only for final reporting. Use the validation split to choose a checkpoint or monitor training.

## Reported U-Net Results

The included epoch-45 checkpoint was evaluated on the 359-image test split with these results:

| Metric | Result |
|---|---:|
| Pixel accuracy | 88.18% |
| Macro F1 | 55.14% |
| Mean IoU | 44.03% |
| Marine Debris recall | 76.00% |
| Marine Debris IoU | 25.00% |

Accuracy is influenced by common water classes. Macro F1 and Mean IoU better reflect performance across rare and common classes. Weighted loss is used to reduce class-imbalance effects, but it does not eliminate the imbalance; some rare classes may still have low recall or IoU.

## Predict One Image

`predict.py` accepts one 11-band Sentinel-2 GeoTIFF and writes a single-band prediction mask:

```bash
python semantic_segmentation/unet/predict.py \
  --input_path path/to/input_patch.tif \
  --model_path semantic_segmentation/unet/trained_models/45/model.pth \
  --output_path data/my_prediction.tif
```

The command prints the number and percentage of pixels predicted as Marine Debris. The input must use the same 11 bands, order, scaling, and preprocessing conventions as MARIDA. An ordinary JPG, screenshot, RGB photograph, or Sentinel-2 RGB preview cannot be passed directly to this model.

## Inspect Results

The prediction output is a georeferenced GeoTIFF. It can be opened in QGIS together with the original image and reference `_cl.tif` mask:

```text
Original image  -> inspect the satellite data
Reference mask  -> ground-truth annotation
Prediction mask -> U-Net result
```

Compare Marine Debris pixels using class value `1`.

## License

This project is intended for academic and research purposes.
