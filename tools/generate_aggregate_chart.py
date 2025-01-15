import pandas as pd
import matplotlib.pyplot as plt

aggregated_file = "results/test_6/aggregated_score_results.xlsx"
output_file = "results/test_6/aggregated_score_with_stats.xlsx"
results_folder = "results/test_6"
EVOLUTIONARY = True
MIN_FITNESS, MAX_FITNESS = 0, 4.5

def plot_mean_std(x_axis, m_vec, std_vec, description, title=None, y_label=None, x_label=None):
    lower_bound = [M_new - Sigma for M_new, Sigma in zip(m_vec, std_vec)]
    upper_bound = [M_new + Sigma for M_new, Sigma in zip(m_vec, std_vec)]

    plt.plot(x_axis, m_vec, '--', label=description + " Avg.")
    plt.fill_between(x_axis, lower_bound, upper_bound, alpha=.3, label=description + " Avg. ± SD")
    if title is not None:
        plt.title(title)
    if y_label is not None:
        plt.ylabel(y_label)
    if x_label is not None:
        plt.xlabel(x_label)

def save_plot_results_fitness(results, results_folder):
    plt.figure()
    plot_mean_std(results['generation'], results['avg_fitness'], results['std_fitness'], "Population")
    plot_mean_std(results['generation'], results['best_avg_fitness'], results['best_std_fitness'], "Bests")
    plt.plot(results['generation'], results['max_fitness'], 'r-', label="Best")
    plt.ylim(MIN_FITNESS, MAX_FITNESS)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid()
    plt.legend()
    plt.savefig(results_folder + "/fitness_evolution.png")

    plt.figure()
    plot_mean_std(results['generation'], results['avg_score'], results['std_score'], "Population")
    plot_mean_std(results['generation'], results['best_avg_score'], results['best_std_score'], "Bests")
    plt.plot(results['generation'], results['max_score'], 'r-', label="Best")
    plt.ylim(0, 10)
    plt.xlabel('Generation')
    plt.ylabel('Aesthetic Score')
    plt.grid()
    plt.legend()
    plt.savefig(results_folder + "/score_evolution.png")

def save_plot_results_score(results, results_folder):
    plt.figure()
    plot_mean_std(results['iteration'], results['average_score'], results['std_score'], "", "Score Evolution")
    plt.plot(results['iteration'], results['max_score'], 'r-', label="Best")
    plt.ylim(0, 15)
    plt.xlabel('Iteration')
    plt.ylabel('Aesthetic Score (SAM)')
    plt.grid()
    plt.legend()
    plt.savefig(results_folder + "/aesthetic_evolution.png")

def save_plot_results_loss(results, results_folder):
    plt.figure()
    plot_mean_std(results['iteration'], results['average_loss'], results['std_loss'], "", "Loss Evolution")
    plt.plot(results['iteration'], results['min_loss'], 'r-', label="Best")
    plt.ylim(0, 1)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    plt.savefig(results_folder + "/loss_evolution.png")

# Load the aggregated data from the Excel file
data = pd.read_excel(aggregated_file)

if EVOLUTIONARY:
    # Calculate the average fitness across all seeds for each iteration
    data['avg_fitness'] = data.filter(like='avg_fitness_').mean(axis=1)

    # Calculate the standard deviation of fitness across all seeds for each iteration
    data['std_fitness'] = data.filter(like='std_fitness_').std(axis=1)

    # Calculate the average fitness across all seeds for each iteration
    data['best_avg_fitness'] = data.filter(like='max_fitness_').mean(axis=1)

    # Calculate the average fitness across all seeds for each iteration
    data['best_std_fitness'] = data.filter(like='max_fitness_').std(axis=1)

    # Calculate the average fitness across all seeds for each iteration
    data['max_fitness'] = data.filter(like='max_fitness_').max(axis=1)

    # Calculate the average fitness across all seeds for each iteration
    data['avg_score'] = data.filter(like='avg_score_').mean(axis=1)

    # Calculate the standard deviation of fitness across all seeds for each iteration
    data['std_score'] = data.filter(like='std_score_').std(axis=1)

    # Calculate the average fitness across all seeds for each iteration
    data['best_avg_score'] = data.filter(like='max_score_').mean(axis=1)

    # Calculate the average fitness across all seeds for each iteration
    data['best_std_score'] = data.filter(like='max_score_').std(axis=1)

    # Calculate the average fitness across all seeds for each iteration
    data['max_score'] = data.filter(like='max_score_').max(axis=1)

    save_plot_results_fitness(data, results_folder)
else:
    # Calculate the average fitness across all seeds for each iteration
    data['average_score'] = data.filter(like='score_').mean(axis=1)

    # Calculate the standard deviation of fitness across all seeds for each iteration
    data['std_score'] = data.filter(like='score_').std(axis=1)

    # Calculate the average fitness across all seeds for each iteration
    data['max_score'] = data.filter(like='score_').max(axis=1)

    # Calculate the average fitness across all seeds for each iteration
    data['average_loss'] = data.filter(like='loss_').mean(axis=1)

    # Calculate the standard deviation of fitness across all seeds for each iteration
    data['std_loss'] = data.filter(like='loss_').std(axis=1)

    # Calculate the average fitness across all seeds for each iteration
    data['min_loss'] = data.filter(like='loss_').min(axis=1)

    save_plot_results_score(data, results_folder)
    save_plot_results_loss(data, results_folder)

# Save the updated data with average and standard deviation columns to a new file
data.to_excel(output_file, index=False)
print(f"Updated data saved to {output_file}")
