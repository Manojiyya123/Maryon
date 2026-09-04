"""Create an RGB preview with predicted Marine Debris highlighted."""

import argparse
import os

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFilter


def stretch_band(band):
    valid = band[np.isfinite(band)]
    if valid.size == 0:
        return np.zeros_like(band, dtype="float32")
    low, high = np.percentile(valid, (2, 98))
    if high <= low:
        return np.zeros_like(band, dtype="float32")
    return np.clip((band - low) / (high - low), 0, 1)


def visualize(options):
    with rasterio.open(options["image_path"]) as source:
        image = source.read().astype("float32")

    with rasterio.open(options["mask_path"]) as source:
        mask = source.read(1)

    ground_truth = None
    if options["ground_truth_path"]:
        with rasterio.open(options["ground_truth_path"]) as source:
            ground_truth = source.read(1)

    if image.shape[0] != 11:
        raise ValueError("Expected an 11-band Sentinel-2 image.")
    if image.shape[1:] != mask.shape:
        raise ValueError("The image and prediction mask must have the same dimensions.")
    if ground_truth is not None and ground_truth.shape != mask.shape:
        raise ValueError("The ground-truth mask must have the same dimensions as the prediction mask.")

    # Sentinel-2 false-color-independent RGB: red=665, green=560, blue=490.
    rgb = np.stack(
        [stretch_band(image[3]), stretch_band(image[2]), stretch_band(image[1])],
        axis=-1,
    )
    debris = mask == 1
    visible_debris = np.array(
        Image.fromarray((debris * 255).astype("uint8"), mode="L").filter(
            ImageFilter.MaxFilter(9)
        )
    ) > 0
    overlay = rgb.copy()
    overlay[visible_debris] = [1.0, 0.0, 0.0]
    debris_mask = np.zeros_like(rgb)
    debris_mask[visible_debris] = [1.0, 0.0, 0.0]

    panels = [rgb]
    panel_names = ["Original"]
    if ground_truth is not None:
        truth_mask = np.zeros_like(rgb)
        truth_mask[ground_truth == 1] = [1.0, 0.0, 0.0]
        panels.append(truth_mask)
        panel_names.append("Ground truth")
    panels.extend([debris_mask, overlay])
    panel_names.extend(["Prediction", "Overlay"])
    preview = np.concatenate(panels, axis=1)

    output_path = options["output_path"]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    preview_image = Image.fromarray((preview * 255).astype("uint8"), mode="RGB")
    draw = ImageDraw.Draw(preview_image)
    panel_width = image.shape[1]
    for index, name in enumerate(panel_names):
        draw.rectangle((index * panel_width, 0, index * panel_width + 100, 22), fill="white")
        draw.text((index * panel_width + 5, 4), name, fill="black")
    preview_image.save(output_path)

    percentage = 100.0 * debris.mean()
    print("Preview saved to: {}".format(output_path))
    print("Marine Debris pixels: {} / {}".format(int(debris.sum()), debris.size))
    print("Marine Debris percentage: {:.2f}%".format(percentage))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True, help="Original 11-band Sentinel-2 GeoTIFF")
    parser.add_argument("--mask_path", required=True, help="Prediction mask GeoTIFF")
    parser.add_argument("--ground_truth_path", default=None, help="Optional ground-truth class mask GeoTIFF")
    parser.add_argument("--output_path", default="../../data/prediction_preview.png")
    visualize(vars(parser.parse_args()))