"""Predict and visualize a randomly selected MARIDA split image."""

import argparse
import os
import random
import subprocess
import sys

import numpy as np


def image_paths(root_path, roi):
    parts = roi.split("_")
    folder = "_".join(["S2"] + parts[:-1])
    name = "_".join(["S2"] + parts)
    image = os.path.join(root_path, "data", "patches", folder, name + ".tif")
    return image, name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model_path", default="trained_models/45/model.pth")
    args = parser.parse_args()

    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    split_path = os.path.join(root_path, "data", "splits", args.split + "_X.txt")
    rois = np.atleast_1d(np.genfromtxt(split_path, dtype="str"))
    random_generator = random.Random(args.seed)
    roi = str(random_generator.choice(rois))
    image_path, image_name = image_paths(root_path, roi)
    mask_path = os.path.join(root_path, "data", "random_prediction.tif")
    ground_truth_path = os.path.join(
        root_path, "data", "patches", image_paths(root_path, roi)[1].rsplit("_", 1)[0],
        image_name + "_cl.tif"
    )
    preview_path = os.path.join(root_path, "data", "random_prediction_preview.png")
    script_dir = os.path.dirname(__file__)
    python = sys.executable

    print("Selected {} image: {}".format(args.split, roi))
    subprocess.run(
        [
            python,
            os.path.join(script_dir, "predict.py"),
            "--input_path",
            image_path,
            "--model_path",
            os.path.join(script_dir, args.model_path),
            "--output_path",
            mask_path,
        ],
        check=True,
    )
    subprocess.run(
        [
            python,
            os.path.join(script_dir, "visualize_prediction.py"),
            "--image_path",
            image_path,
            "--mask_path",
            mask_path,
            "--ground_truth_path",
            ground_truth_path,
            "--output_path",
            preview_path,
        ],
        check=True,
    )
    os.startfile(preview_path)
    print("Original image: {}".format(image_path))
    print("Prediction mask: {}".format(mask_path))
    print("Preview: {}".format(preview_path))