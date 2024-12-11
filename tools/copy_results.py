import os
import shutil

# Define source directory containing the result folders
source_dir = 'results_embedding_opt_adam'
# Define destination directory for the copied folders
destination_dir = 'results_embedding_opt_adam_copy'

# List of specific files to copy
files_to_copy = [
    "it_1.png",
    "it_10000.png",
    "it_20000.png",
    "it_30000.png",
    "it_40000.png",
    "it_50000.png",
    "best_all.png",
    "fitness_evolution.png",
    "fitness_results.csv"
]

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Iterate over all folders in the source directory
for folder_name in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder_name)
    if os.path.isdir(folder_path) and folder_name.startswith("results_LAION_"):
        # Create the corresponding folder in the destination directory
        dest_folder_path = os.path.join(destination_dir, folder_name)
        os.makedirs(dest_folder_path, exist_ok=True)
        
        # Copy the specified files if they exist
        for file_name in files_to_copy:
            src_file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(src_file_path):
                dest_file_path = os.path.join(dest_folder_path, file_name)
                shutil.copy(src_file_path, dest_file_path)
                print(f"Copied {src_file_path} to {dest_file_path}")
            else:
                print(f"File {file_name} not found in {folder_path}")

print("Copy process completed.")
