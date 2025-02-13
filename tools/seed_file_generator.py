import os

# Define the seeds and number of files
seeds = [
    42, 101, 256, 333, 512, 789, 1001, 1200, 1500, 1707,
    1984, 2023, 3000, 4096, 5012, 6789, 7001, 8096, 9009, 10007,
    11011, 12121, 13013, 14141, 15015, 16061, 17171, 18181, 19191, 20021
]
num_files = 3  # Adjust this number to change the number of output files

# Calculate the number of seeds per file
seeds_per_file = len(seeds) // num_files

# Create files and write seeds
for i in range(num_files):
    # Determine the start and end indices for the seeds in the current file
    start_idx = i * seeds_per_file
    end_idx = (i + 1) * seeds_per_file if i < num_files - 1 else len(seeds)

    # Get the subset of seeds for the current file
    seeds_subset = seeds[start_idx:end_idx]

    # Write the seeds to the file
    os.makedirs('seeds', exist_ok=True)
    with open(f'seeds/seeds_{i + 1}.txt', 'w') as file:
        for seed in seeds_subset:
            file.write(f"{seed}\n")