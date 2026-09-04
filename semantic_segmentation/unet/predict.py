"""Predict marine-debris classes for one Sentinel-2 GeoTIFF patch."""

import argparse
import os

import numpy as np
import rasterio
import torch
import torchvision.transforms as transforms

from dataloader import bands_mean, bands_std
from unet import UNet


def predict(options):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        input_bands=options["input_channels"],
        output_classes=options["output_channels"],
        hidden_channels=options["hidden_channels"],
    ).to(device)
    model.load_state_dict(torch.load(options["model_path"], map_location=device))
    model.eval()

    with rasterio.open(options["input_path"]) as source:
        image = source.read().astype("float32")
        metadata = source.meta.copy()

    if image.shape[0] != options["input_channels"]:
        raise ValueError(
            "Expected {} image bands, found {}.".format(
                options["input_channels"], image.shape[0]
            )
        )
    if image.shape[1] % 16 != 0 or image.shape[2] % 16 != 0:
        raise ValueError("Image height and width must be divisible by 16.")

    image = np.moveaxis(image, 0, -1)
    nan_mask = np.isnan(image)
    image[nan_mask] = np.tile(
        bands_mean, (image.shape[0], image.shape[1], 1)
    )[nan_mask]

    image = transforms.ToTensor()(image)
    image = transforms.Normalize(bands_mean, bands_std)(image)

    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))
        predicted_classes = logits.argmax(1).squeeze(0).cpu().numpy() + 1

    output_path = options["output_path"]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    metadata.update(count=1, dtype="uint8")
    with rasterio.open(output_path, "w", **metadata) as destination:
        destination.write(predicted_classes.astype("uint8"), 1)

    debris_pixels = int(np.count_nonzero(predicted_classes == 1))
    total_pixels = predicted_classes.size
    debris_percentage = 100.0 * debris_pixels / total_pixels
    print("Prediction saved to: {}".format(output_path))
    print("Marine Debris pixels: {} / {}".format(debris_pixels, total_pixels))
    print("Marine Debris percentage: {:.2f}%".format(debris_percentage))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="Input 11-band Sentinel-2 GeoTIFF")
    parser.add_argument("--model_path", required=True, help="Trained U-Net checkpoint")
    parser.add_argument("--output_path", default="../../data/prediction.tif")
    parser.add_argument("--input_channels", type=int, default=11)
    parser.add_argument("--output_channels", type=int, default=11)
    parser.add_argument("--hidden_channels", type=int, default=16)
    predict(vars(parser.parse_args()))