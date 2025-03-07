import os
import pandas as pd

# Directory containing the result folders
base_dir = "results/cmaes_embedding_laion_partial_40"
output_file = "results/cmaes_embedding_laion_partial_40/aggregated_score_results.xlsx"

EVOLUTIONARY = True
PREDICTOR = "LAION"

# Initialize aggregated_data as None
aggregated_data = None

# Iterate over all subdirectories
for folder_name in os.listdir(base_dir):
    if folder_name.startswith(f"results_{PREDICTOR}_"):
        seed = folder_name.split("_")[-1]  # Extract the seed number

        if EVOLUTIONARY:
            file_path = os.path.join(base_dir, folder_name, "fitness_results.csv")
        else:
            file_path = os.path.join(base_dir, folder_name, "score_results.csv")
        
        if os.path.exists(file_path):
            # Read the CSV file
            df = pd.read_csv(file_path)

            if not EVOLUTIONARY:
                df = df.drop(columns=["mean_grad","total_grad_norm"])
            
            # Ensure the iteration column exists
            if not EVOLUTIONARY and "iteration" not in df.columns:
                raise ValueError(f"Missing 'iteration' column in {file_path}")
            
            if EVOLUTIONARY and "generation" not in df.columns:
                raise ValueError(f"Missing 'generation' column in {file_path}")
            
            if EVOLUTIONARY:
                df = df.rename(columns={"avg_fitness": f"avg_fitness_{seed}"})
                df = df.rename(columns={"max_fitness": f"max_fitness_{seed}"})
                df = df.rename(columns={"std_fitness": f"std_fitness_{seed}"})
                df = df.rename(columns={"avg_score": f"avg_score_{seed}"})
                df = df.rename(columns={"max_score": f"max_score_{seed}"})
                df = df.rename(columns={"std_score": f"std_score_{seed}"})
                df = df.rename(columns={"elapsed_time": f"elapsed_time_{seed}"})
            else:
                df = df.rename(columns={"score": f"score_{seed}"}) 
                df = df.rename(columns={"loss": f"loss_{seed}"})  
                df = df.rename(columns={"elapsed_time": f"elapsed_time_{seed}"})
            
            # Merge data into the aggregated data
            if aggregated_data is None:
                aggregated_data = df
            elif EVOLUTIONARY:
                aggregated_data = pd.merge(aggregated_data, df, on="generation", how="outer")
            else:
                aggregated_data = pd.merge(aggregated_data, df, on="iteration", how="outer")
        else:
            print(f"File not found: {file_path}")

# Ensure aggregated_data is not None before saving
if aggregated_data is not None:
    # Save the aggregated data to an Excel file
    aggregated_data.to_excel(output_file, index=False)
    print(f"Aggregated results saved to {output_file}")
else:
    print("No data was aggregated. Check the input folders and files.")
