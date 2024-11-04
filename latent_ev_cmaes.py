import torch
import numpy as np
import pandas as pd
from diffusers import UNet2DModel, DDIMScheduler, VQModel
import cma
import random
from PIL import Image
import simulacra_rank_image
import laion_rank_image
import nima_rank_image
import argparse
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Receives argument seed (int).')

parser.add_argument('--seed', type=int, help='Seed')
parser.add_argument('--seed_path', type=str, help='Path to seed list file')
parser.add_argument('--cuda', type=int, help='Cuda GPU to use')
parser.add_argument('--predictor', type=int, help='Aesthetic predictor to use\n0 - SAM\n1 - LAION\n2 - NIMA')

args = parser.parse_args()

if args.seed_path is not None:
    SEED_PATH = args.seed_path
elif args.seed is not None:
    print("Seed path not provided, executing single run")
    SEED = args.seed
    SEED_PATH = None
else:
    print("Seed path not provided, executing single run")
    print("Seed not provided, default is 42")
    SEED = 42
    SEED_PATH = None

if args.cuda is not None:
    cuda_n = str(args.cuda)
else:
    print("Cuda device not provided, default is 0")
    cuda_n = str(0)

if args.predictor is not None:
    predictor = args.predictor
else:
    print("Aesthetic predictor not provided, default is 0 (SAM)")
    predictor = 0

NUM_GENERATIONS, POP_SIZE = 100, 100  # Adjust as needed

if SEED_PATH is None:
    seed_list = [SEED]
else:
    with open(SEED_PATH, 'r') as file:
        # Read each line, strip newline characters, and convert to integers
        seed_list = [int(line.strip()) for line in file]

# Check if a GPU is available and if not, use the CPU
device = "cuda:" + cuda_n if torch.cuda.is_available() else "cpu"

# Load all models
unet = UNet2DModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="unet")
vqvae = VQModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="vqvae")
scheduler = DDIMScheduler.from_pretrained("CompVis/ldm-celebahq-256", subfolder="scheduler")

unet.to(device)
vqvae.to(device)

num_inference_steps = 25
guidance_scale = 7.5

# Define the scheduler
scheduler.set_timesteps(num_inference_steps)

if predictor == 1:
    aesthetic_model = laion_rank_image.LAIONAesthetic(device)
elif predictor == 2:
    aesthetic_model = nima_rank_image.NIMAAsthetics()
else:
    aesthetic_model = simulacra_rank_image.SimulacraAesthetic(device)

# Define the aesthetic evaluation function
def aesthetic_evaluation(image):
    if predictor == 0:
        aesthetic_score = aesthetic_model.predict_from_pil(image)
    elif predictor == 1 or predictor == 2:
        aesthetic_score = aesthetic_model.predict(image)
    else:
        # Other metrics here
        return 0
    return aesthetic_score.item()

# Function to generate an image from noise vector
def generate_image_from_noise(noise):
    image = noise
    for t in scheduler.timesteps:
        # predict noise residual of previous image
        with torch.no_grad():
            residual = unet(image, t)["sample"]

        # compute previous image x_t according to DDIM formula
        prev_image = scheduler.step(residual, t, image, eta=0.0)["prev_sample"]

        # x_t-1 -> x_t
        image = prev_image

    # decode image with vae
    with torch.no_grad():
        image = vqvae.decode(image).sample

    # process image
    image_processed = image.permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.clamp(0, 255).cpu().numpy().astype(np.uint8)
    image_processed = image_processed[0]  # Remove batch dimension

    # Convert to PIL Image
    pil_image = Image.fromarray(image_processed)
    return pil_image

def evaluate(x):
    # x is a NumPy array representing the noise vector
    # Convert it to a torch tensor
    noise = torch.tensor(x, dtype=torch.float32, device=device)
    # Reshape the noise to the original shape
    noise = noise.view(1, unet.in_channels, unet.sample_size, unet.sample_size)
    image = generate_image_from_noise(noise)
    score = aesthetic_evaluation(image)
    # CMA-ES minimizes the function, so we need to invert the score if higher is better
    return -score  # Negate the score to turn maximization into minimization

def normalize_noise_vector(x):
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std == 0:
        x_std = 1e-8  # Prevent division by zero
    x_normalized = (x - x_mean) / x_std
    return x_normalized


def main(seed):
    generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    noise = torch.randn(
        (1, unet.in_channels, unet.sample_size, unet.sample_size),
        generator=generator,
    ).to(device)
    noise_shape = noise.shape
    noise_size = noise.numel()
    initial_noise = noise.cpu().numpy().flatten()

    results_folder = "results_" + str(predictor) + "_" + str(seed)

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Set CMA-ES options
    es_options = {
        'seed': seed,
        'popsize': POP_SIZE,
        'maxiter': NUM_GENERATIONS,
        'verb_filenameprefix': results_folder + '/outcmaes',  # Save logs
        'verb_log': 0,  # Disable log output
        'verbose': -9,  # Suppress console output
    }

    # Initialize CMA-ES
    sigma0 = 0.5  # Initial standard deviation
    es = cma.CMAEvolutionStrategy(initial_noise, sigma0, es_options)

    max_fit_list = []
    avg_fit_list = []
    std_fit_list = []
    best_list = []

    generation = 0

    while not es.stop():
        print(f"Generation {generation+1}/{NUM_GENERATIONS}")
        # Ask for new candidate solutions
        solutions = es.ask()
        # Normalize each solution
        solutions = [normalize_noise_vector(x) for x in solutions]
        # Evaluate candidate solutions
        fitnesses = []
        for x in solutions:
            fitness = evaluate(x)
            fitnesses.append(fitness)
        # Tell CMA-ES the fitnesses
        es.tell(solutions, fitnesses)

        # Record statistics
        scores = [-f for f in fitnesses]  # Convert back to positive scores
        max_fit = max(scores)
        avg_fit = np.mean(scores)
        std_fit = np.std(scores)

        max_fit_list.append(max_fit)
        avg_fit_list.append(avg_fit)
        std_fit_list.append(std_fit)

        # Get best solution so far
        best_x = es.result.xbest
        best_score = -es.result.fbest  # Convert back to positive score
        best_list.append(best_score)

        # Generate and save the best image
        best_noise = torch.tensor(best_x, dtype=torch.float32, device=device)
        best_noise = best_noise.view(noise_shape)
        best_image = generate_image_from_noise(best_noise)
        best_image.save(results_folder + "/best_%d.png" % (generation+1))

        # Print stats
        print(f"Generation {generation+1}: Max fitness: {max_fit}, Avg fitness: {avg_fit}")

        generation += 1
        if generation >= NUM_GENERATIONS:
            break

    # Save the overall best image
    best_noise = torch.tensor(es.result.xbest, dtype=torch.float32, device=device)
    best_noise = best_noise.view(noise_shape)
    best_image = generate_image_from_noise(best_noise)
    best_image.save(results_folder + "/best_all.png")

    # Save the metrics
    results = pd.DataFrame({
        "generation": list(range(1, generation + 1)),
        "best_fitness": max_fit_list,
        "average_fitness": avg_fit_list,
        "std_fitness": std_fit_list
    })

    results.to_csv(results_folder + "/fitness_results.csv", index=False)

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

def save_plot_results(results, results_folder):
    plt.figure()
    plot_mean_std(results['generation'], results['average_fitness'], results['std_fitness'], "Population")
    plt.plot(results['generation'], results['best_fitness'], 'r-', label="Best")
    plt.ylim(0, 10)
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Aesthetic Score)')
    plt.grid()
    plt.legend()
    plt.savefig(results_folder + "/fitness_evolution.png")

if __name__ == "__main__":
    for seed in seed_list:
        main(seed)
        print(f"Run with seed {seed} finished!")
