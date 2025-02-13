import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

aggregated_file_1 = "results/adam_embedding_laion/aggregated_score_results.xlsx"
aggregated_file_2 = "results/test_4/aggregated_score_results.xlsx"
aggregated_file_3 = "results/test_hybrid/aggregated_score_results.xlsx"
results_folder = "results/test_hybrid"
EVOLUTIONARY = [False, True, True]
MIN_FITNESS, MAX_FITNESS = 0, 12
MIN_SCORE, MAX_SCORE = 0, 12

APPROACH_NAMES = ["Adam", "CMA-ES", "Hybrid"]

def normalize_dataframes_to_smallest(*dataframes):
    # Find the minimum length of all dataframes
    min_length = min(len(df) for df in dataframes)
    
    # Interpolate each dataframe to match the minimum length
    i = 0
    normalized_dataframes = []
    for df in dataframes:
        # Use linear interpolation to match the size
        interpolated_df = df.reindex(
            np.linspace(0, len(df) - 1, min_length)
        ).interpolate(method='linear').reset_index(drop=True)
        
        if EVOLUTIONARY[i]:
            interpolated_df['generation'] = np.linspace(0, 100, min_length)
        else:
            interpolated_df['iteration'] = np.linspace(0, 100, min_length)
        
        normalized_dataframes.append(interpolated_df)
        i += 1
    
    return normalized_dataframes

def plot_mean_std(x_axis, m_vec, std_vec, description, title=None, y_label=None, x_label=None):
    lower_bound = [M_new - Sigma for M_new, Sigma in zip(m_vec, std_vec)]
    upper_bound = [M_new + Sigma for M_new, Sigma in zip(m_vec, std_vec)]

    plt.plot(x_axis, m_vec, label=description)
    #plt.fill_between(x_axis, lower_bound, upper_bound, alpha=.3)
    if title is not None:
        plt.title(title)
    if y_label is not None:
        plt.ylabel(y_label)
    if x_label is not None:
        plt.xlabel(x_label)

def save_plot_results_score_comparison(results_1, results_2, results_3, approach_names, results_folder):
    plt.figure()

    if EVOLUTIONARY[0]:
        plot_mean_std(results_1['generation'], results_1['best_avg_score'], results_1['best_std_score'], approach_names[0], "")
    else:
        plot_mean_std(results_1['iteration'], results_1['average_score'], results_1['std_score'], approach_names[0], "")

    if EVOLUTIONARY[1]:
        plot_mean_std(results_2['generation'], results_2['best_avg_score'], results_2['best_std_score'], approach_names[1], "")
    else:
        plot_mean_std(results_2['iteration'], results_2['average_score'], results_2['std_score'], approach_names[1], "")

    if EVOLUTIONARY[2]:
        plot_mean_std(results_3['generation'], results_3['best_avg_score'], results_3['best_std_score'], approach_names[2], "")
    else:
        plot_mean_std(results_3['iteration'], results_3['average_score'], results_3['std_score'], approach_names[2], "")
    

    plt.ylim(0, 12)
    plt.xlabel('Iteration (%)')
    plt.ylabel('Aesthetic Score (LAION)')
    plt.grid()
    plt.legend()
    plt.savefig(results_folder + "/aesthetic_evolution_comparison.png")

    plt.figure()

    if EVOLUTIONARY[0]:
        plot_mean_std(results_1['generation'], results_1['elapsed_time'], [], approach_names[0], "")
    else:
        plot_mean_std(results_1['iteration'], results_1['elapsed_time'], [], approach_names[0], "")

    if EVOLUTIONARY[1]:
        plot_mean_std(results_2['generation'], results_2['elapsed_time'], [], approach_names[1], "")
    else:
        plot_mean_std(results_2['iteration'], results_2['elapsed_time'], [], approach_names[1], "")

    if EVOLUTIONARY[2]:
        plot_mean_std(results_3['generation'], results_3['elapsed_time'], [], approach_names[2], "")
    else:
        plot_mean_std(results_3['iteration'], results_3['elapsed_time'], [], approach_names[2], "")   

    plt.xlabel('Iteration (%)')
    plt.ylabel('Elapsed Time (Seconds)')
    plt.grid()
    plt.legend()
    plt.savefig(results_folder + "/elapsed_time_comparison.png")


# Load the aggregated data from the Excel file
data_1 = pd.read_excel(aggregated_file_1)
data_2 = pd.read_excel(aggregated_file_2)
data_3 = pd.read_excel(aggregated_file_3)

if EVOLUTIONARY[0]:
    # Calculate the average fitness across all seeds for each iteration
    data_1['best_avg_score'] = data_1.filter(like='max_score_').mean(axis=1)

    # Calculate the average fitness across all seeds for each iteration
    data_1['best_std_score'] = data_1.filter(like='max_score_').std(axis=1)
else:
    # Calculate the average fitness across all seeds for each iteration
    data_1['average_score'] = data_1.filter(like='score_').mean(axis=1)

    # Calculate the standard deviation of fitness across all seeds for each iteration
    data_1['std_score'] = data_1.filter(like='score_').std(axis=1)

if EVOLUTIONARY[1]:
    # Calculate the average fitness across all seeds for each iteration
    data_2['best_avg_score'] = data_2.filter(like='max_fitness_').mean(axis=1)

    # Calculate the average fitness across all seeds for each iteration
    data_2['best_std_score'] = data_2.filter(like='max_fitness_').std(axis=1)
else:
    # Calculate the average fitness across all seeds for each iteration
    data_2['average_score'] = data_2.filter(like='score_').mean(axis=1)

    # Calculate the standard deviation of fitness across all seeds for each iteration
    data_2['std_score'] = data_2.filter(like='score_').std(axis=1)

if EVOLUTIONARY[2]:
    # Calculate the average fitness across all seeds for each iteration
    data_3['best_avg_score'] = data_3.filter(like='max_score_').mean(axis=1)

    # Calculate the average fitness across all seeds for each iteration
    data_3['best_std_score'] = data_3.filter(like='max_score_').std(axis=1)
else:
    # Calculate the average fitness across all seeds for each iteration
    data_3['average_score'] = data_3.filter(like='score_').mean(axis=1)

    # Calculate the standard deviation of fitness across all seeds for each iteration
    data_3['std_score'] = data_3.filter(like='score_').std(axis=1)

data_1['elapsed_time'] = data_1.filter(like='elapsed_time_').mean(axis=1)
data_2['elapsed_time'] = data_2.filter(like='elapsed_time_').mean(axis=1)
data_3['elapsed_time'] = data_3.filter(like='elapsed_time_').mean(axis=1)

data_1, data_2, data_3 = normalize_dataframes_to_smallest(data_1, data_2, data_3)

save_plot_results_score_comparison(data_1, data_2, data_3, APPROACH_NAMES, results_folder)

print(f"Adam avg score: {data_1['average_score'].iloc[-1]}")
print(f"Adam std score: {data_1['std_score'].iloc[-1]}")
print(f"Adam avg time: {data_1['elapsed_time'].iloc[-1]}")
print(f"CMA-ES avg score: {data_2['best_avg_score'].iloc[-1]}")
print(f"CMA-ES std score: {data_2['best_std_score'].iloc[-1]}")
print(f"CMA-ES avg time: {data_2['elapsed_time'].iloc[-1]}")
print(f"Hybrid avg score: {data_3['best_avg_score'].iloc[-1]}")
print(f"Hybrid std score: {data_3['best_std_score'].iloc[-1]}")
print(f"Hybrid avg time: {data_3['elapsed_time'].iloc[-1]}")