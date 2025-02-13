import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Define source directories containing the result folders
source_dirs = [
    'results/adam_embedding_laion',
    'results/test_4',
    'results/test_hybrid'
]
# Output file for the grid
output_grid_path = 'results/test_hybrid/best_all_grid_compare.png'

EVOLUTIONARY, SHOW_FITNESS = True, False
PREDICTOR = "LAION"

# Grid dimensions
x_rows = 30
y_columns = 4  # 1 column for it_0.png and 3 columns for best_all.png from each source directory
max_images = x_rows * y_columns

# Initialize list for image paths
image_paths = []

# Iterate over folders in the first source directory
for folder_name in os.listdir(source_dirs[0]):
    folder_paths = [os.path.join(source_dir, folder_name) for source_dir in source_dirs]
    
    if all(os.path.isdir(folder_path) for folder_path in folder_paths):
        # Locate the it_0.png image and best_all.png images
        it_0_path = os.path.join(folder_paths[0], "it_0.png")
        best_all_paths = [os.path.join(folder_path, "best_all.png") for folder_path in folder_paths]
        
        if os.path.isfile(it_0_path) and all(os.path.isfile(best_all_path) for best_all_path in best_all_paths):
            image_paths.append([it_0_path] + best_all_paths)

# Select only the top images that fit into the grid
image_paths = image_paths[:x_rows]

# Create the grid
fig, axes = plt.subplots(x_rows, y_columns, figsize=(y_columns * 3, x_rows * 3))
axes = axes.flatten()

for i, ax in enumerate(axes):
    row = i // y_columns
    col = i % y_columns
    
    if row < len(image_paths):
        image_path = image_paths[row][col]
        
        # Load the image
        img = Image.open(image_path)
        
        # Display the image
        ax.imshow(img)
        ax.axis("off")
        
        # Annotate with the image type
        if row == 0 and col == 0:
            ax.set_title("Initial Embedding", fontsize=20)
        elif row == 0 and col == 1:
            ax.set_title(f"Adam", fontsize=20)
        elif row == 0 and col == 2:
            ax.set_title(f"CMA-ES", fontsize=20)
        elif row == 0 and col == 3:
            ax.set_title(f"Hybrid", fontsize=20)
    else:
        # Turn off unused subplots
        ax.axis("off")

# Adjust layout and save the grid
plt.tight_layout()
plt.savefig(output_grid_path)
print(f"Grid saved to {output_grid_path}")
