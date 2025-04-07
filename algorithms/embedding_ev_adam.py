import sys
import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from diffusers import StableDiffusionPipeline, DDIMScheduler
import random
from PIL import Image
import argparse
import matplotlib.pyplot as plt
import time

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to sys.path to obtain access to the submodules
sys.path.insert(0, parent_dir)

# Aesthetic Evaluators
import src.nima_rank_image as nima_rank_image
import src.simulacra_rank_image as simulacra_rank_image
import src.laion_rank_image as laion_rank_image

# Half Precision
from torch.cuda.amp import autocast, GradScaler

# Load configuration from YAML file
config_path = "algorithms/config/config_adam.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

SEED = config['seed']
SEED_PATH = config['seed_path']
cuda_n = str(config['cuda'])
predictor = config['predictor']
num_inference_steps = config['num_inference_steps']
guidance_scale = config['guidance_scale']
height = config['height']
width = config['width']
NUM_ITERATIONS = config['num_iterations']
model_id = config['model_id']
results_folder = config['results_folder']
learning_rate = float(config['learning_rate'])
weight_decay = float(config['weight_decay'])
epsilon = float(config['epsilon'])

# Check if a GPU is available and if not, use the CPU
device = "cuda:" + cuda_n if torch.cuda.is_available() else "cpu"

# Load the Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
pipe.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe.to(device)

# Define the scheduler
pipe.scheduler.set_timesteps(num_inference_steps)

# Initialize the aesthetic model
if predictor == 0:
    # Simulacra Aesthetic Model (SAM)
    aesthetic_model = simulacra_rank_image.SimulacraAesthetic(device)
    model_name = 'SAM'
    # Ensure that SAM supports backpropagation
    supports_grad = True  # Set to True if SAM supports backpropagation
elif predictor == 1:
    # LAION Aesthetic Predictor
    aesthetic_model = laion_rank_image.LAIONAesthetic(device)
    model_name = 'LAION'
    # Ensure that LAION supports backpropagation
    supports_grad = True  # Set to True if LAION supports backpropagation
elif predictor == 2:
    # NIMA
    aesthetic_model = nima_rank_image.NIMAAesthetics(device)
    model_name = 'NIMA'
    supports_grad = True
else:
    raise ValueError("Invalid predictor option.")

if not supports_grad:
    raise ValueError(f"The selected aesthetic model '{model_name}' does not support backpropagation required for gradient-based optimization.")

if SEED_PATH is None:
    seed_list = [SEED]
else:
    with open(SEED_PATH, 'r') as file:
        # Read each line, strip newline characters, and convert to integers
        seed_list = [int(line.strip()) for line in file]

def generate_image_from_embeddings(text_embeddings, seed):
    # Create unconditional embeddings for classifier-free guidance
    uncond_input = pipe.tokenizer(
        "",
        return_tensors="pt",
        padding="max_length",
        max_length=77,
        truncation=True
    ).to(device)
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0]
    # uncond_embeddings are in float32

    # Concatenate the unconditional and text embeddings
    encoder_hidden_states = torch.cat([uncond_embeddings, text_embeddings], dim=0)

    # Fix the initial latents
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)  # Use the seed
    latents = torch.randn(
        (1, pipe.unet.config.in_channels, height // 8, width // 8),
        generator=generator,
        device=device,
        dtype=torch.float32  # Use float16 for latents
    )

    # Denoising loop
    for i, t in enumerate(pipe.scheduler.timesteps):
        # Expand the latents if we are doing classifier-free guidance
        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents

        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states)["sample"]

        # Perform guidance
        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Compute the previous noisy sample x_t -> x_t-1
        latents = pipe.scheduler.step(noise_pred, t, latents)["prev_sample"]

    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor)["sample"]
    image = (image / 2 + 0.5).clamp(0, 1)  # Normalize to [0,1]
    image = image.squeeze(0).permute(1, 2, 0)  # Convert to [H, W, C]

    return image  # Return as tensor

def aesthetic_evaluation(image):
    # image is a tensor of shape [H, W, C]
    # Convert to [N, C, H, W] and ensure it's in float32
    image_input = image.permute(2, 0, 1).to(torch.float32)  # [1, C, H, W]

    if predictor == 0:
        # Simulacra Aesthetic Model
        score = aesthetic_model.predict_from_tensor(image_input)
    elif predictor == 1:
        # LAION Aesthetic Predictor
        score = aesthetic_model.predict_from_tensor(image_input)
    elif predictor == 2:
        # NIMA
        score = aesthetic_model.predict(image_input)
    else:
        return torch.tensor(0.0, device=device)

    return score

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

def main(seed, seed_number):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    results_folder_seed = f"{results_folder}/results_{model_name}_{seed}"
    os.makedirs(results_folder_seed, exist_ok=True)

    # Initialize the text embeddings with an empty prompt
    text_input = pipe.tokenizer(
        "",
        return_tensors="pt",
        padding="max_length",
        max_length=77,
        truncation=True
    ).to(device)
    text_embeddings_init = pipe.text_encoder(text_input.input_ids.to(device))[0]
    text_embeddings = torch.nn.Parameter(text_embeddings_init.clone())

    with torch.no_grad():
        initial_image = generate_image_from_embeddings(text_embeddings_init, seed)
        image_np = initial_image.detach().clone().cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        pil_image.save(f"{results_folder_seed}/it_0.png")

        initial_score = aesthetic_evaluation(initial_image)
        initial_loss = 1/(1+initial_score)  # Negative because we want to maximize the score

    score_list = [initial_score.item()]
    loss_list = [initial_loss.item()]
    time_list = [0]
    best_score = initial_score
    best_text_embeddings = text_embeddings_init.detach().clone()

    mean_grad_list = [0]
    total_norm_list = [0]

    optimizer = torch.optim.Adam([text_embeddings], lr=learning_rate, weight_decay=weight_decay, eps=epsilon)

    scaler = GradScaler()  # Initialize GradScaler for mixed precision training

    start_time = time.time()

    for iteration in range(1, NUM_ITERATIONS + 1):
        print(f"Iteration {iteration}/{NUM_ITERATIONS}")

        optimizer.zero_grad()

        with autocast(dtype=torch.float16):
            image = generate_image_from_embeddings(text_embeddings, seed)
            score = aesthetic_evaluation(image)
            loss = 1/(1+score)  # Negative because we want to maximize the score

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        mean_grad = text_embeddings.grad.mean()
        total_norm = 0
        if text_embeddings.grad is not None:
            param_norm = text_embeddings.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        mean_grad_list.append(mean_grad.cpu().numpy())
        total_norm_list.append(total_norm)

        if score.item() > best_score:
            best_score = score.item()
            best_text_embeddings = text_embeddings.detach().clone()

        score_list.append(score.item())
        loss_list.append(loss.item())

        image_np = image.detach().clone().cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        pil_image.save(f"{results_folder_seed}/it_{iteration}.png")

        elapsed_time = time.time() - start_time
        iterations_done = iteration
        iterations_left = NUM_ITERATIONS - iteration
        average_time_per_iteration = elapsed_time / iterations_done
        estimated_time_remaining = average_time_per_iteration * iterations_left

        formatted_time_remaining = format_time(estimated_time_remaining)

        time_list.append(elapsed_time)

        # Save the metrics
        results = pd.DataFrame({
            "iteration": list(range(0, iteration + 1)),
            "score": score_list,
            "loss": loss_list,
            "mean_grad": mean_grad_list,
            "total_grad_norm": total_norm_list,
            "elapsed_time": time_list
        })

        results.to_csv(f"{results_folder_seed}/score_results.csv", index=False, na_rep='nan')

        # Plot and save the fitness evolution
        plot_results(results, results_folder_seed)

        # Print stats
        print(f"Seed {seed_number} Iteration {iteration}/{NUM_ITERATIONS}: Score: {score.item()}, Estimated time remaining: {formatted_time_remaining}")

    # Save the overall best image
    with torch.no_grad():
        best_image = generate_image_from_embeddings(best_text_embeddings, seed)
    best_image_np = best_image.detach().cpu().numpy()
    best_image_np = (best_image_np * 255).astype(np.uint8)
    pil_image = Image.fromarray(best_image_np)
    pil_image.save(f"{results_folder_seed}/best_all.png")

def plot_results(results, results_folder_seed):
    plt.figure()
    plt.plot(results['iteration'], results['score'], label="Score")
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.title('Score over Iterations')
    plt.grid()
    plt.legend()
    plt.savefig(results_folder_seed + "/score_evolution.png")
    plt.close()

    plt.figure()
    plt.plot(results['iteration'], results['loss'], label="Loss")
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss over Iterations')
    plt.grid()
    plt.legend()
    plt.savefig(results_folder_seed + "/loss_evolution.png")
    plt.close()

if __name__ == "__main__":
    i = 1
    for seed in seed_list:
        main(seed, i)
        print(f"Run with seed {seed} finished!")
        i += 1