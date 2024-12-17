import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

# Define source directory containing the result folders
source_dir = 'results_embedding_opt_adam_copy'
# Output file for the grid
output_grid_path = 'results_embedding_opt_adam_copy/best_all_grid.png'

# Grid dimensions
x_rows = 6
y_columns = 5

# Initialize lists for seeds, scores, and image paths
seed_info = []

# Iterate over folders
for folder_name in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder_name)
    if os.path.isdir(folder_path) and folder_name.startswith("results_LAION_"):
        # Extract the seed number from the folder name
        seed_number = folder_name.split("_")[-1]
        
        # Locate the image and CSV file
        image_path = os.path.join(folder_path, "best_all.png")
        csv_path = os.path.join(folder_path, "fitness_results.csv")
        
        if os.path.isfile(image_path) and os.path.isfile(csv_path):
            # Read the score from the CSV file
            try:
                df = pd.read_csv(csv_path)
                max_score = df["score"].max()
                seed_info.append((seed_number, max_score, image_path))
            except Exception as e:
                print(f"Error reading CSV for {folder_name}: {e}")
        else:
            print(f"Missing required files in {folder_name}")

# Sort by seed number (optional)
seed_info.sort(key=lambda x: int(x[0]))

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
        ax.set_title(f"Seed: {seed}\nScore: {score:.2f}", fontsize=10)
    else:
        # Turn off unused subplots
        ax.axis("off")

# Adjust layout and save the grid
plt.tight_layout()
plt.savefig(output_grid_path)
print(f"Grid saved to {output_grid_path}")
