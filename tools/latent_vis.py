import os
import glob
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import json
import matplotlib.pyplot as plt

# Path to the CSV files
best_csv_files_path = 'results/cmaes_embedding_laion_01sigma/results_LAION_*/best_fitness_embeddings.csv'
gen_csv_files_path = 'results/cmaes_embedding_laion_01sigma/results_LAION_*/gen_*/fitness_embeddings.csv'
results_path = 'results/cmaes_embedding_laion_01sigma'

# Aggregate data from all CSV files
best_files = glob.glob(best_csv_files_path)
gen_files = glob.glob(gen_csv_files_path)
data_list = []

for file in best_files + gen_files:
    df = pd.read_csv(file)
    # Convert the JSON strings in the 'best_embeddings' or 'embeddings' column to lists
    if 'best_embeddings' in df.columns:
        df['embeddings'] = df['best_embeddings']
    df['embeddings'] = df['embeddings'].apply(json.loads)
    data_list.append(df)

# Concatenate all dataframes
data = pd.concat(data_list, ignore_index=True)

# Extract embeddings and scores
embeddings = np.vstack(data['embeddings'].values)
if 'max_fitness' in data.columns:
    scores = data['max_fitness'].values
else:
    scores = data['fitnesses'].values

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot the results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=scores, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Score')
plt.title('t-SNE Visualization of Embeddings')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.savefig(results_path + "/embeddings_tsne.png")