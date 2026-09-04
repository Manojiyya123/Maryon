# Local Data Folders

The MARIDA dataset is not committed because it is large. Download it separately and place it here.

Required structure:

```text
data/
├── patches/
│   └── S2_DATE_TILE/
│       ├── S2_DATE_TILE_CROP.tif
│       ├── S2_DATE_TILE_CROP_cl.tif
│       └── S2_DATE_TILE_CROP_conf.tif
├── splits/
│   ├── train_X.txt
│   ├── val_X.txt
│   └── test_X.txt
└── predicted_unet/
```

The training and prediction scripts use paths relative to this `data` folder.