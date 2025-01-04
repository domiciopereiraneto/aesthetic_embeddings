import pandas as pd
import matplotlib.pyplot as plt

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

def save_plot_results_score(results, results_folder):
    plt.figure()
    plot_mean_std(results['iteration'], results['average_score'], results['std_score'], "", "Score Evolution")
    plt.plot(results['iteration'], results['max_score'], 'r-', label="Best")
    plt.ylim(0, 15)
    plt.xlabel('Iteration')
    plt.ylabel('Aesthetic Score')
    plt.grid()
    plt.legend()
    plt.savefig(results_folder + "/aesthetic_evolution.png")

def save_plot_results_loss(results, results_folder):
    plt.figure()
    plot_mean_std(results['iteration'], results['average_loss'], results['std_loss'], "", "Loss Evolution")
    plt.plot(results['iteration'], results['min_loss'], 'r-', label="Best")
    plt.ylim(0, 1)
    plt.xlabel('Iteration')
    plt.ylabel('Aesthetic Loss')
    plt.grid()
    plt.legend()
    plt.savefig(results_folder + "/loss_evolution.png")

# Load the aggregated data from the Excel file
aggregated_file = "results/test/aggregated_score_results.xlsx"
data = pd.read_excel(aggregated_file)

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

save_plot_results_score(data, "results/test_2")
save_plot_results_loss(data, "results/test_2")

# Plot the average fitness with error bars (standard deviation)
# plt.figure(figsize=(10, 6))
# plt.errorbar(
#     data['iteration'],
#     data['average_score'],
#     yerr=data['std_score'],
#     fmt='o-',
#     ecolor='red',
#     capsize=3,
#     label='Average Score ± Std. Dev'
# )
# plt.title('Average Aesthetic Score (SAM)')
# plt.xlabel('Iteration')
# plt.ylabel('Score')
# plt.grid(True)
# plt.legend()
# plt.savefig("results_adam_SAM_opt/aggregated_aesthetic_score_evolution.png")

# Save the updated data with average and standard deviation columns to a new file
output_file = "results/test_2/aggregated_score_with_stats.xlsx"
data.to_excel(output_file, index=False)
print(f"Updated data saved to {output_file}")
