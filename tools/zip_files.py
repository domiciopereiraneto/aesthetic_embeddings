import os
import glob
import zipfile

# Define the pattern to match the files
pattern = 'results/cmaes_embedding_laion_01sigma/results_LAION_*/best_fitness_embeddings.csv'

# Find all files matching the pattern
files_to_zip = glob.glob(pattern)

# Define the name of the output zip file
zip_filename = 'best_fitness_embeddings.zip'

# Create a zip file and add each file to it
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for file in files_to_zip:
        zipf.write(file, os.path.basename(file))

print(f'Created {zip_filename} containing {len(files_to_zip)} files.')