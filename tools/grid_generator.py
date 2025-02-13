import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Define source directory containing the result folders
source_dir = 'results/test_hybrid'
# Output file for the grid
output_grid_path = 'results/test_hybrid/best_all_grid.png'

EVOLUTIONARY, SHOW_FITNESS = True, False
PREDICTOR = "LAION"

# Grid dimensions
x_rows = 5
y_columns = 6
max_images = x_rows * y_columns

# Initialize lists for seeds, scores, and image paths
seed_info = []

# Iterate over folders
for folder_name in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder_name)
    if os.path.isdir(folder_path) and folder_name.startswith(f"results_{PREDICTOR}_"):
        # Extract the seed number from the folder name
        seed_number = folder_name.split("_")[-1]
        
        # Locate the image and CSV file
        image_path = os.path.join(folder_path, "best_all.png")

        if EVOLUTIONARY:
            csv_path = os.path.join(folder_path, "fitness_results.csv")
        else:
            csv_path = os.path.join(folder_path, "score_results.csv")
        
        if os.path.isfile(image_path) and os.path.isfile(csv_path):
            # Read the score from the CSV file
            try:
                df = pd.read_csv(csv_path)
                if EVOLUTIONARY and SHOW_FITNESS:
                    max_score = df["max_fitness"].max()
                elif EVOLUTIONARY:
                    max_score = df["max_score"].max()
                else:
                    max_score = df["score"].max()
                seed_info.append((seed_number, max_score, image_path))
            except Exception as e:
                print(f"Error reading CSV for {folder_name}: {e}")
        else:
            print(f"Missing required files in {folder_name}")

# Sort by score or fitness in descending order
seed_info.sort(key=lambda x: float(x[1]), reverse=True)

# Select only the top images that fit into the grid
seed_info = seed_info[:max_images]

# Create the grid
fig, axes = plt.subplots(x_rows, y_columns, figsize=(y_columns * 3, x_rows * 3))
axes = axes.flatten()

for i, ax in enumerate(axes):
    if i < len(seed_info):
        seed, score, image_path = seed_info[i]
        
        # Load the image
        img = Image.open(image_path)
        
        # Display the image
        ax.imshow(img)
        ax.axis("off")
        
        # Annotate with seed and score
        if EVOLUTIONARY and SHOW_FITNESS:
            ax.set_title(f"Seed: {seed}\nFitness: {score:.2f}", fontsize=10)
        else:
            ax.set_title(f"Seed: {seed}\nScore: {score:.2f}", fontsize=10)
    else:
        # Turn off unused subplots
        ax.axis("off")

# Adjust layout and save the grid
plt.tight_layout()
plt.savefig(output_grid_path)
print(f"Grid saved to {output_grid_path}")
