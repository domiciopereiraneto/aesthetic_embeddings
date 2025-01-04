import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to sys.path to obtain access to the submolues
sys.path.insert(0, parent_dir)

import os
import re
from PIL import Image
import pandas as pd
import numpy as np
import torch

import nima_rank_image
import simulacra_rank_image
import laion_rank_image

device = "cuda:2"
models = {"SAM": True, "Laion": True, "Nima": True}

if models["SAM"]:
    # Simulacra Aesthetic Model (SAM)
    model_sam = simulacra_rank_image.SimulacraAesthetic(device)
if models["Laion"]:
    # LAION Aesthetic Predictor
    model_laion = laion_rank_image.LAIONAesthetic(device)
if models["Nima"]:
    # NIMA
    model_nima = nima_rank_image.NIMAAsthetics()

# Function to process images in a folder
def process_images_in_folder(folder_path):
    score_sam_list = []
    score_laion_list = []
    score_nima_list = []
    file_name_list = []
    image_pattern = re.compile(r"it_\d+\.png")  # Regex for file names like it_x.png
    image_files = [f for f in sorted(os.listdir(folder_path)) if image_pattern.match(f)]
    total_images = len(image_files)  # Get total number of images to process
    processed_count = 0  # Initialize processed images count

    for file_name in image_files:
        image_path = os.path.join(folder_path, file_name)
        try:
            # Load and transform the image
            pil_image = Image.open(image_path).convert("RGB")
            image_np = np.array(pil_image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)
            image_input = image_tensor.permute(2, 0, 1).to(torch.float32).to(device)

            file_name_list.append(file_name)

            if models["SAM"]:
                # Simulacra Aesthetic Model (SAM)
                score_sam = model_sam.predict_from_tensor(image_input)
                score_sam_list.append(score_sam.item())
            if models["Laion"]:
                # LAION Aesthetic Predictor
                score_laion = model_laion.predict_from_tensor(image_input)
                score_laion_list.append(score_laion.item())
            if models["Nima"]:
                # NIMA
                score_nima = model_nima.predict_from_tensor(image_input)
                score_nima_list.append(score_nima.item())

            # Increment and print progress
            processed_count += 1
            print(f"Processed {processed_count}/{total_images} images...")

            # Save predictions to a CSV file
            if len(file_name_list) > 0:
                data = {"file_name": file_name_list}
                if len(score_sam_list) > 0:
                    data["sam_score"] = score_sam_list
                if len(score_laion_list) > 0:
                    data["laion_score"] = score_laion_list
                if len(score_nima_list) > 0:
                    data["nima_score"] = score_nima_list
                csv_path = os.path.join(folder_path, "predictions.csv")
                df = pd.DataFrame(data=data)
                df.to_csv(csv_path, index=False)
                #print(f"Saved predictions to {csv_path}")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

# Iterate over subfolders and process images
def process_subfolders(root_folder):
    subfolders = [
        subfolder_name for subfolder_name in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, subfolder_name)) and subfolder_name.startswith("results_SAM_")
    ]
    total_subfolders = len(subfolders)  # Get total number of subfolders
    processed_count = 0  # Initialize processed subfolders count

    for subfolder_name in subfolders:
        subfolder_path = os.path.join(root_folder, subfolder_name)
        try:
            print(f"Processing folder: {subfolder_name} ({processed_count + 1}/{total_subfolders})")
            process_images_in_folder(subfolder_path)
            processed_count += 1
            print(f"Finished processing folder: {subfolder_name} ({processed_count}/{total_subfolders})")
        except Exception as e:
            print(f"Error processing folder {subfolder_name}: {e}")

# Replace with the root folder containing subfolders
root_folder = "results/test"

# Run the script
process_subfolders(root_folder)
