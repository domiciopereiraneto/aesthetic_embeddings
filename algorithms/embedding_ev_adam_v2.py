import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from diffusers import StableDiffusionPipeline, DDIMScheduler
import random
from PIL import Image
# Import your aesthetic models here
import nima_rank_image
import simulacra_rank_image
import laion_rank_image
import argparse
import os
import matplotlib.pyplot as plt
import time
from torch.cuda.amp import autocast, GradScaler  # Import AMP utilities

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
    cuda_n = str(0)

if args.predictor is not None:
    predictor = args.predictor
else:
    print("Aesthetic predictor not provided, default is 0 (SAM)")
    predictor = 0  # Set default to SAM

num_inference_steps = 30
guidance_scale = 10
height = 256
width = 256

NUM_ITERATIONS = 100  

# Check if a GPU is available and if not, use the CPU
device = "cuda:" + cuda_n if torch.cuda.is_available() else "cpu"

# Load the Stable Diffusion pipeline
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
pipe.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe.to(device)

# Set text_encoder to float32 for better precision
pipe.text_encoder.to(torch.float32)

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

def generate_image_from_embeddings(text_embeddings_initial, seed):
    # Ensure text_embeddings are in float32
    text_embeddings_initial = text_embeddings_initial.to(torch.float32)

    text_tokens_padding = pipe.tokenizer(
        "",
        return_tensors="pt",
        padding="max_length",
        max_length=67,
        truncation=True
    ).to(device)
    text_embeddings_padding = pipe.text_encoder(text_tokens_padding.input_ids.to(device))[0]

    text_embeddings = torch.cat([text_embeddings_initial, text_embeddings_padding], dim=1)

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
    # Cast encoder_hidden_states to float16 for the UNet
    encoder_hidden_states = encoder_hidden_states.to(torch.float16)

    # Fix the initial latents
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)  # Use the seed
    latents_input = torch.randn(
        (1, pipe.unet.config.in_channels, height // 8, width // 8),
        generator=generator,
        device=device,
        dtype=torch.float16  # Use float16 for latents
    )

    # Denoising loop
    for i, t in enumerate(pipe.scheduler.timesteps):
        # Expand latents if using guidance
        latent_model_input = (
            torch.cat([latents_input] * 2)
            if guidance_scale > 1.0 else latents_input
        )

        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict noise residual
        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=encoder_hidden_states
        )["sample"]

        # Perform guidance
        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Compute the previous noisy sample x_t -> x_t-1
        latents_input = pipe.scheduler.step(noise_pred, t, latents_input)["prev_sample"]

    # Decode the image
    image = pipe.vae.decode(latents_input / pipe.vae.config.scaling_factor)["sample"]
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

    # Initialize the text embeddings with an empty prompt
    text_input = pipe.tokenizer(
        "pretty sunset",
        return_tensors="pt",
        padding="max_length",
        max_length=10,
        truncation=True
    ).to(device)
    text_embeddings_init = pipe.text_encoder(text_input.input_ids.to(device))[0]
    text_embeddings = torch.nn.Parameter(text_embeddings_init.clone())

    optimizer = torch.optim.Adam([text_embeddings], lr=1e-3, weight_decay=0, eps=1e-8)  # Adjust learning rate as needed

    scaler = GradScaler()  # Initialize the GradScaler for AMP

    results_folder = f"results_embedding_opt_adam/results_{model_name}_{seed}"
    os.makedirs(results_folder, exist_ok=True)

    max_fit_list = []
    best_score = -float('inf')
    best_text_embeddings = None

    mean_grad_list = []
    # std_grad_list = []
    # max_grad_list = []
    # min_grad_list = []
    total_norm_list = []

    start_time = time.time()

    for iteration in range(1, NUM_ITERATIONS + 1):
        print(f"Iteration {iteration}/{NUM_ITERATIONS}")

        optimizer.zero_grad()

        with autocast():  # Enable autocasting
            image = generate_image_from_embeddings(text_embeddings, seed)
            score = aesthetic_evaluation(image)
            loss = 1/(1+score)  # Negative because we want to maximize the score
            #loss = score/10

        scaler.scale(loss).backward()  # Scale the loss and backward pass

        mean_grad = text_embeddings.grad.mean()
        # std_grad = text_embeddings.grad.std()
        # max_grad = text_embeddings.grad.max()
        # min_grad = text_embeddings.grad.min()
        total_norm = 0
        if text_embeddings.grad is not None:
            param_norm = text_embeddings.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        mean_grad_list.append(mean_grad.cpu().numpy())
        total_norm_list.append(total_norm)

        torch.nn.utils.clip_grad_norm_([text_embeddings], max_norm=1.0)
        torch.nn.utils.clip_grad_value_([text_embeddings], 10)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            # Normalize text_embeddings to prevent gradient explosion
            text_embeddings.copy_((text_embeddings - text_embeddings.mean()) / text_embeddings.std())
            if torch.any(torch.isnan(text_embeddings)):
                print("Gradient explosion detected.")
                break

        if score.item() > best_score:
            best_score = score.item()
            best_text_embeddings = text_embeddings.detach().clone()

        max_fit_list.append(score.item())

        image_np = image.detach().clone().cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        pil_image.save(f"{results_folder}/it_{iteration}.png")

        elapsed_time = time.time() - start_time
        iterations_done = iteration
        iterations_left = NUM_ITERATIONS - iteration
        average_time_per_iteration = elapsed_time / iterations_done
        estimated_time_remaining = average_time_per_iteration * iterations_left

        formatted_time_remaining = format_time(estimated_time_remaining)

        # Print stats
        print(f"Seed {seed_number} Iteration {iteration}/{NUM_ITERATIONS}: Score: {score.item()}, Estimated time remaining: {formatted_time_remaining}")

    # Save the overall best image
    with torch.no_grad(), autocast():
        best_image = generate_image_from_embeddings(best_text_embeddings, seed)
    best_image_np = best_image.detach().cpu().numpy()
    best_image_np = (best_image_np * 255).astype(np.uint8)
    pil_image = Image.fromarray(best_image_np)
    pil_image.save(f"{results_folder}/best_all.png")

    # Save the metrics
    results = pd.DataFrame({
        "iteration": list(range(1, NUM_ITERATIONS + 1)),
        "score": max_fit_list,
        "mean_grad": mean_grad_list,
        "total_grad_norm": total_norm_list
    })

    results.to_csv(f"{results_folder}/fitness_results.csv", index=False, na_rep='nan')

    # Plot and save the fitness evolution
    plot_results(results, results_folder)

def plot_results(results, results_folder):
    plt.figure()
    plt.plot(results['iteration'], results['score'], label="Score")
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.title('Score over Iterations')
    plt.grid()
    plt.legend()
    plt.savefig(results_folder + "/fitness_evolution.png")
    plt.close()

if __name__ == "__main__":
    i = 1
    for seed in seed_list:
        main(seed, i)
        print(f"Run with seed {seed} finished!")
        i += 1
