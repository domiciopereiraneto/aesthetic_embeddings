import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to sys.path to obtain access to the submolues
sys.path.insert(0, parent_dir)

import torch
import numpy as np
import pandas as pd
from diffusers import StableDiffusionPipeline
import random
from PIL import Image
import argparse
import sys
import os
import matplotlib.pyplot as plt
import time

# Aesthetic Evaluators
import nima_rank_image
import simulacra_rank_image
import laion_rank_image

parser = argparse.ArgumentParser(description='Receives argument seed (int).')

parser.add_argument('--seed', type=int, help='Seed')
parser.add_argument('--seed_path', type=str, help='Path to seed list file')
parser.add_argument('--cuda', type=int, help='CUDA GPU to use')
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
    print("CUDA device not provided, default is 0")
    cuda_n = str(2)

if args.predictor is not None:
    predictor = args.predictor
else:
    print("Aesthetic predictor not provided, default is 0 (SAM)")
    predictor = 1  # Set default to SAM

NUM_SAMPLES = 10  # Number of random samples to generate
VECTOR_SIZE = 15  # Size of the token vector
OUTPUT_FOLDER = "results/random_sampling"

# Check if a GPU is available and if not, use the CPU
device = "cuda:" + cuda_n if torch.cuda.is_available() else "cpu"

# Load the Stable Diffusion pipeline
model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to(device)

num_inference_steps = 50
guidance_scale = 7.5

# Define the scheduler
pipe.scheduler.set_timesteps(num_inference_steps)

# Initialize the aesthetic model
if predictor == 0:
    aesthetic_model = simulacra_rank_image.SimulacraAesthetic(device)
    model_name = 'SAM'
elif predictor == 1:
    aesthetic_model = laion_rank_image.LAIONAesthetic(device)
    model_name = 'LAION'
elif predictor == 2:
    aesthetic_model = nima_rank_image.NIMAAesthetics(device)
    model_name = 'NIMA'
else:
    raise ValueError("Invalid predictor option.")

if SEED_PATH is None:
    seed_list = [SEED]
else:
    with open(SEED_PATH, 'r') as file:
        # Read each line, strip newline characters, and convert to integers
        seed_list = [int(line.strip()) for line in file]

MIN_VALUE, MAX_VALUE = 0, pipe.tokenizer.vocab_size - 3
START_OF_TEXT, END_OF_TEXT = pipe.tokenizer.bos_token_id, pipe.tokenizer.eos_token_id

# Create unconditional embeddings for classifier-free guidance
uncond_input = pipe.tokenizer("", return_tensors="pt", padding="max_length", max_length=77, truncation=True).to(device)
uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0]

def aesthetic_evaluation(image):
    # image is a tensor of shape [H, W, C]
    # Convert to [N, C, H, W] and ensure it's in float32
    image_input = image.permute(2, 0, 1).to(torch.float32)  # [1, C, H, W]

    if predictor == 0:
        # Simulacra Aesthetic Model
        score = aesthetic_model.predict_from_tensor(image_input)
    elif predictor == 1:
        # LAION Aesthetic Predictor
        score = aesthetic_model.predict_from_tensor(image_input)[0].item()
    elif predictor == 2:
        # NIMA
        score = aesthetic_model.predict(image_input)
    else:
        return torch.tensor(0.0, device=device)

    return score

def generate_image_from_embeddings(token_vector, num_inference_steps=25):
    tmp_token_vector = np.insert(token_vector.cpu().detach().numpy().flatten(), 0, START_OF_TEXT)
    padding_size = pipe.tokenizer.model_max_length - len(tmp_token_vector.flatten())
    tmp_token_vector = np.append(tmp_token_vector, [END_OF_TEXT] * padding_size)
    tmp_token_vector = torch.tensor(tmp_token_vector, dtype=torch.int64).to(device)
    tmp_token_vector = torch.clamp(tmp_token_vector, MIN_VALUE, MAX_VALUE).view(1, len(tmp_token_vector)).to(device)
    text_embeddings = pipe.text_encoder(tmp_token_vector)[0]

    # Concatenate the unconditional and text embeddings
    encoder_hidden_states = torch.cat([uncond_embeddings, text_embeddings])

    tmp_latents = torch.randn((1, pipe.unet.config.in_channels, 512 // 8, 512 // 8), device=device)

    # Denoising loop
    for i, t in enumerate(pipe.scheduler.timesteps):
        latent_model_input = torch.cat([tmp_latents] * 2) if guidance_scale > 1.0 else tmp_latents
        with torch.no_grad():
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states)["sample"]
        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        tmp_latents = pipe.scheduler.step(noise_pred, t, tmp_latents)["prev_sample"]

    with torch.no_grad():
        image = pipe.vae.decode(tmp_latents / pipe.vae.config.scaling_factor)["sample"]
    image = (image / 2 + 0.5).clamp(0, 1)  # Normalize to [0,1]
    image = image.squeeze(0).permute(1, 2, 0)  # Convert to [H, W, C]
    
    return image

def main(seed, seed_number):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    results_folder = f"{OUTPUT_FOLDER}/results_{model_name}_{seed}"
    os.makedirs(results_folder, exist_ok=True)

    max_score_list = []
    best_list = []
    prompt_list = []

    time_list = []

    best_score_overall = -float('inf')
    best_token_vector_overall = None
    best_image = None

    start_time = time.time()

    for i in range(NUM_SAMPLES):
        token_vector = torch.randint(MIN_VALUE, MAX_VALUE, (VECTOR_SIZE,), device=device)
        image = generate_image_from_embeddings(token_vector)
        score = aesthetic_evaluation(image)

        if score > best_score_overall:
            best_score_overall = score
            best_token_vector_overall = token_vector
            best_image = image

        max_score_list.append(best_score_overall)
        best_list.append(best_token_vector_overall.cpu().numpy())
        prompt_list.append(detokenize(token_vector))

        # Save the overall best image
        image_np = best_image.detach().clone().cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        pil_image = Image.fromarray((image_np))
        pil_image.save(f"{results_folder}/best_{i+1}.png")

        elapsed_time = time.time() - start_time
        iterations_done = i + 1
        iterations_left = NUM_SAMPLES - iterations_done
        average_time_per_iteration = elapsed_time / iterations_done
        estimated_time_remaining = average_time_per_iteration * iterations_left

        formatted_time_remaining = format_time(estimated_time_remaining)

        time_list.append(elapsed_time)

        print(f"Sample {i + 1}/{NUM_SAMPLES}: Max score: {max_score_list[-1]}, Estimated time remaining: {formatted_time_remaining}")

    results = pd.DataFrame({
        "sample": list(range(1, NUM_SAMPLES + 1)),
        "score": max_score_list,
        "best_individual": best_list,
        "elapsed_time": time_list,
        "prompt": prompt_list
    })

    results.to_csv(f"{results_folder}/score_results.csv", index=False)

    save_plot_results(results, results_folder)

    print(f"Run with seed {seed} finished!")

def detokenize(individual):
    tmp_solution = torch.tensor(individual, dtype=torch.int64)
    tmp_solution = torch.clamp(tmp_solution, 0, pipe.tokenizer.vocab_size - 1)
    decoded_string = pipe.tokenizer.decode(tmp_solution, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return decoded_string

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
    plt.plot(results['sample'], results['score'], 'r-')
    plt.ylim(0, 10)
    plt.xlabel('Iteration')
    plt.ylabel('Aesthetic Score')
    plt.grid()
    plt.legend()
    plt.savefig(results_folder + "/score_evolution.png")

def format_time(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

if __name__ == "__main__":
    i = 1
    for seed in seed_list:
        main(seed, i)
        print(f"Run with seed {seed} finished!")
        i += 1