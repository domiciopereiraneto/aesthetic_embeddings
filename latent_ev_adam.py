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

def aesthetic_evaluation(image_tensor):
    # image_tensor: [batch_size, height, width, channels]
    image_tensor = image_tensor.permute(0, 3, 1, 2)  # Convert to [batch_size, channels, height, width]

    if predictor == 0:
        # Simulacra Aesthetic Model
        aesthetic_score = aesthetic_model.predict(image_tensor)
    elif predictor == 1:
        # LAION Aesthetic Model
        aesthetic_score = aesthetic_model.predict(image_tensor)
    elif predictor == 2:
        # NIMA Aesthetic Model
        aesthetic_score = aesthetic_model.predict(image_tensor)
    else:
        return torch.tensor(0.0, device=device)

    return aesthetic_score

def generate_image_from_noise(noise):
    image = noise
    for t in scheduler.timesteps:
        # Predict noise residual of previous image
        residual = unet(image, t)["sample"]

        # Compute previous image x_t according to DDIM formula
        prev_image = scheduler.step(residual, t, image, eta=0.0)["prev_sample"]

        # x_t-1 -> x_t
        image = prev_image

    # Decode image with VQ-VAE
    image = vqvae.decode(image).sample

    # Process image
    image_processed = image.permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.clamp(0, 255) / 255.0  # Normalize to [0,1]

    return image_processed  # Return as tensor

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


def main_adam(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    noise = torch.randn(
        (1, unet.in_channels, unet.sample_size, unet.sample_size),
        device=device,
        requires_grad=True
    )

    results_folder = "results_adam_" + str(predictor) + "_" + str(seed)
    os.makedirs(results_folder, exist_ok=True)

    optimizer = torch.optim.Adam([noise], lr=1e-2)

    max_fit_list = []
    best_score = -float('inf')
    best_noise = None

    NUM_ITERATIONS = 100  # Adjust as needed

    for iteration in range(1, NUM_ITERATIONS + 1):
        print(f"Iteration {iteration}/{NUM_ITERATIONS}")

        optimizer.zero_grad()

        image = generate_image_from_noise(noise)
        score = aesthetic_evaluation(image)
        loss = -score  # Negative because we want to maximize the score

        loss.backward()
        optimizer.step()

        # Optionally normalize or clip noise
        with torch.no_grad():
            noise_mean = noise.mean()
            noise_std = noise.std()
            if noise_std == 0:
                noise_std = 1e-8
            noise.data = (noise - noise_mean) / noise_std

        if score.item() > best_score:
            best_score = score.item()
            best_noise = noise.detach().clone()

        max_fit_list.append(score.item())

        # Save the best image
        best_image_tensor = generate_image_from_noise(best_noise)
        best_image = best_image_tensor.detach().cpu().numpy()[0]
        best_image = (best_image * 255).astype(np.uint8)
        best_image = Image.fromarray(best_image)
        best_image.save(f"{results_folder}/best_{iteration}.png")

        # Print stats
        print(f"Iteration {iteration}: Score: {score.item()}")

    # Save the overall best image
    best_image_tensor = generate_image_from_noise(best_noise)
    best_image = best_image_tensor.detach().cpu().numpy()[0]
    best_image = (best_image * 255).astype(np.uint8)
    best_image = Image.fromarray(best_image)
    best_image.save(f"{results_folder}/best_all.png")

    # Save the metrics
    results = pd.DataFrame({
        "iteration": list(range(1, NUM_ITERATIONS + 1)),
        "score": max_fit_list,
    })

    results.to_csv(f"{results_folder}/fitness_results.csv", index=False)

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
        main_adam(seed)
        print(f"Run with seed {seed} finished!")
