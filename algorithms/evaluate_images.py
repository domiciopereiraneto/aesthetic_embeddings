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

import nima_rank_image
import simulacra_rank_image
import laion_rank_image

device = "cuda:2"
models = {"SAM": False, "Laion": True, "Nima": True}

if models["SAM"]:
    # Simulacra Aesthetic Model (SAM)
    model_sam = simulacra_rank_image.SimulacraAesthetic(device)
if models["Laion"]:
    # LAION Aesthetic Predictor
    model_laion = laion_rank_image.LAIONAesthetic(device)
if models["Nima"]:
    # NIMA
    model_nima = nima_rank_image.NIMAAesthetics()

# Function to process images in a folder
def process_images_in_folder(folder_path):
    score_sam_list = []
    score_laion_list = []
    score_nima_list = []
    file_name_list = []
    image_pattern = re.compile(r"it_\d+\.png")  # Regex for file names like it_x.png
    for file_name in sorted(os.listdir(folder_path)):
        if image_pattern.match(file_name):  # Check if file matches pattern
            image_path = os.path.join(folder_path, file_name)
            try:
                # Load and transform the image
                image = Image.open(image_path).convert("RGB")

                file_name_list.append(file_name)

                if models["SAM"]:
                    # Simulacra Aesthetic Model (SAM)
                    score_sam = model_sam.predict_from_pil(image)
                    score_sam_list.append(score_sam)
                if models["Laion"]:
                    # LAION Aesthetic Predictor
                    score_laion = model_laion.predict_from_pil(image)
                    score_laion_list.append(score_laion)
                if models["Nima"]:
                    # NIMA
                    score_nima = model_nima.predict(image)
                    score_nima_list.append(score_nima)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

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
        print(f"Saved predictions to {csv_path}")

# Iterate over subfolders and process images
def process_subfolders(root_folder):
    for subfolder_name in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder_name)
        if os.path.isdir(subfolder_path) and subfolder_path.startswith("results_SAM_"):  # Check if it's a directory
            print(f"Processing folder: {subfolder_name}")
            process_images_in_folder(subfolder_path)

# Replace with the root folder containing subfolders
root_folder = "results/test"

# Run the script
process_subfolders(root_folder)
