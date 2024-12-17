import os
import pandas as pd

# Directory containing the result folders
base_dir = "results_embedding_opt_adam_copy"
output_file = "results_embedding_opt_adam_copy/aggregated_score_results.xlsx"

# Initialize aggregated_data as None
aggregated_data = None

# Iterate over all subdirectories
for folder_name in os.listdir(base_dir):
    if folder_name.startswith("results_LAION_"):
        seed = folder_name.split("_")[-1]  # Extract the seed number
        file_path = os.path.join(base_dir, folder_name, "fitness_results.csv")
        
        if os.path.exists(file_path):
            # Read the CSV file
            df = pd.read_csv(file_path)

            df = df.drop(columns=["mean_grad","total_grad_norm"])
            
            # Ensure the iteration column exists
            if "iteration" not in df.columns:
                raise ValueError(f"Missing 'iteration' column in {file_path}")
            
            # Rename fitness column to include the seed
            df = df.rename(columns={"score": f"score_{seed}"})  # Replace "score" with "fitness" if needed
            
            # Merge data into the aggregated data
            if aggregated_data is None:
                aggregated_data = df
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
