import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# List of aggregate file paths and corresponding settings
aggregated_files = [
    "results/cmaes_embedding_laion_partial_10/aggregated_score_results.xlsx",
    "results/cmaes_embedding_laion_partial_20/aggregated_score_results.xlsx",
    "results/cmaes_embedding_laion_partial_30/aggregated_score_results.xlsx",
    "results/cmaes_embedding_laion_partial_40/aggregated_score_results.xlsx"
]
results_folder = "results/cmaes_embedding_laion_partial_40"
# Each entry indicates whether the respective approach is evolutionary (True for 'generation', False for 'iteration')
EVOLUTIONARY = [True, True, True, True]
MIN_FITNESS, MAX_FITNESS = 0, 12
MIN_SCORE, MAX_SCORE = 0, 12
APPROACH_NAMES = ["10 tokens", "20 tokens", "30 tokens", "40 tokens"]

def normalize_dataframes_to_smallest(*dataframes):
    # Find the minimum length of all dataframes
    min_length = min(len(df) for df in dataframes)
    
    normalized_dataframes = []
    for i, df in enumerate(dataframes):
        # Use linear interpolation to match the size
        interpolated_df = df.reindex(
            np.linspace(0, len(df) - 1, min_length)
        ).interpolate(method='linear').reset_index(drop=True)
        
        if EVOLUTIONARY[i]:
            interpolated_df['generation'] = np.linspace(0, 100, min_length)
        else:
            interpolated_df['iteration'] = np.linspace(0, 100, min_length)
        
        normalized_dataframes.append(interpolated_df)
    
    return normalized_dataframes

def plot_mean_std(x_axis, m_vec, std_vec, description, title=None, y_label=None, x_label=None):
    lower_bound = [m - s for m, s in zip(m_vec, std_vec)]
    upper_bound = [m + s for m, s in zip(m_vec, std_vec)]
    plt.plot(x_axis, m_vec, label=description)
    # Uncomment the next line if you prefer to show the spread
    # plt.fill_between(x_axis, lower_bound, upper_bound, alpha=0.3)
    if title is not None:
        plt.title(title)
    if y_label is not None:
        plt.ylabel(y_label)
    if x_label is not None:
        plt.xlabel(x_label)

def save_plot_results_score_comparison(results_list, approach_names, evolutionary_flags, results_folder):
    # Plot Score Evolution
    plt.figure()
    for i, df in enumerate(results_list):
        if evolutionary_flags[i]:
            x = df['generation']
            y_mean = df['best_avg_score']
            y_std = df['best_std_score']
        else:
            x = df['iteration']
            y_mean = df['average_score']
            y_std = df['std_score']
        plot_mean_std(x, y_mean, y_std, approach_names[i])
    plt.ylim(0, 12)
    plt.xlabel('Iteration (%)')
    plt.ylabel('Aesthetic Score (LAION)')
    plt.grid()
    plt.legend()
    plt.savefig(results_folder + "/aesthetic_evolution_comparison.png")

    # Plot Elapsed Time Comparison
    plt.figure()
    for i, df in enumerate(results_list):
        if evolutionary_flags[i]:
            x = df['generation']
        else:
            x = df['iteration']
        y = df['elapsed_time']
        # Empty std vector since it's not used here
        plot_mean_std(x, y, [0]*len(y), approach_names[i])
    plt.xlabel('Iteration (%)')
    plt.ylabel('Elapsed Time (Seconds)')
    plt.grid()
    plt.legend()
    plt.savefig(results_folder + "/elapsed_time_comparison.png")

# --- Main execution ---
# Load the aggregated data
dataframes = [pd.read_excel(file) for file in aggregated_files]

# Process each dataframe based on whether it is evolutionary or not
for i, df in enumerate(dataframes):
    if EVOLUTIONARY[i]:
        # Compute mean and std for max scores
        df['best_avg_score'] = df.filter(like='max_score_').mean(axis=1)
        df['best_std_score'] = df.filter(like='max_score_').std(axis=1)
    else:
        df['average_score'] = df.filter(like='score_').mean(axis=1)
        df['std_score'] = df.filter(like='score_').std(axis=1)
    # Compute elapsed time across seeds
    df['elapsed_time'] = df.filter(like='elapsed_time_').mean(axis=1)

# Normalize dataframes to the smallest length
normalized_dataframes = normalize_dataframes_to_smallest(*dataframes)

# Generate plots for score and time comparisons
save_plot_results_score_comparison(normalized_dataframes, APPROACH_NAMES, EVOLUTIONARY, results_folder)

# --- Print summary statistics for each approach ---
for i, df in enumerate(normalized_dataframes):
    if EVOLUTIONARY[i]:
        final_avg = df['best_avg_score'].iloc[-1]
        final_std = df['best_std_score'].iloc[-1]
    else:
        final_avg = df['average_score'].iloc[-1]
        final_std = df['std_score'].iloc[-1]
    final_time = df['elapsed_time'].iloc[-1]
    print(f"{APPROACH_NAMES[i]}: Avg Score: {final_avg}, Std Score: {final_std}, Avg Time: {final_time}")