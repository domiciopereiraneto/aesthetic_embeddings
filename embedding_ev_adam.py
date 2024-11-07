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

# Check if a GPU is available and if not, use the CPU
device = "cuda:" + cuda_n if torch.cuda.is_available() else "cpu"

# Load the Stable Diffusion pipeline
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
pipe.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe.to(device)

num_inference_steps = 30
guidance_scale = 7.5

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

NUM_ITERATIONS = 1000  # Adjust as needed

if SEED_PATH is None:
    seed_list = [SEED]
else:
    with open(SEED_PATH, 'r') as file:
        # Read each line, strip newline characters, and convert to integers
        seed_list = [int(line.strip()) for line in file]

# Height and width of the images
height = 256
width = 256

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

    # Concatenate the unconditional and text embeddings
    encoder_hidden_states = torch.cat([uncond_embeddings, text_embeddings], dim=0)

    # Fix the initial latents
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)  # Use the seed
    latents_input = torch.randn(
        (1, pipe.unet.config.in_channels, height // 8, width // 8),
        generator=generator,
        device=device,
        dtype=torch.float32
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
    # Depending on the model, adjust preprocessing
    image_input = image.permute(2, 0, 1)  # [1, C, H, W]

    if predictor == 0:
        # Simulacra Aesthetic Model
        score = aesthetic_model.predict_from_tensor(image_input)
    elif predictor == 1:
        # LAION Aesthetic Predictor
        score = aesthetic_model.predict(image_input)
    elif predictor == 2:
        # NIMA
        score = aesthetic_model.predict(image_input)
    else:
        return torch.tensor(0.0, device=device)

    return score

def main(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

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

    optimizer = torch.optim.Adam([text_embeddings], lr=1e-3, weight_decay=1e-5)  # Adjust learning rate as needed

    results_folder = f"results_adam_{model_name}_{seed}"
    os.makedirs(results_folder, exist_ok=True)

    max_fit_list = []
    best_score = -float('inf')
    best_text_embeddings = None

    for iteration in range(1, NUM_ITERATIONS + 1):
        print(f"Iteration {iteration}/{NUM_ITERATIONS}")

        optimizer.zero_grad()

        image = generate_image_from_embeddings(text_embeddings, seed)
        score = aesthetic_evaluation(image)
        loss = -score  # Negative because we want to maximize the score

        loss.backward()
        torch.nn.utils.clip_grad_norm_([text_embeddings], max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            norm = text_embeddings.norm(p=2, dim=-1, keepdim=True)
            text_embeddings.copy_(text_embeddings / norm.clamp(min=1e-8))

        if score.item() > best_score:
            best_score = score.item()
            best_text_embeddings = text_embeddings.detach().clone()

        max_fit_list.append(score.item())

        with torch.no_grad():
            best_image = generate_image_from_embeddings(best_text_embeddings, seed)
        best_image_np = best_image.detach().cpu().numpy()
        best_image_np = (best_image_np * 255).astype(np.uint8)
        pil_image = Image.fromarray(best_image_np)
        pil_image.save(f"{results_folder}/best_{iteration}.png")

        # Print stats
        print(f"Iteration {iteration}: Score: {score.item()}")

    # Save the overall best image
    with torch.no_grad():
        best_image = generate_image_from_embeddings(best_text_embeddings, seed)
    best_image_np = best_image.detach().cpu().numpy()
    best_image_np = (best_image_np * 255).astype(np.uint8)
    pil_image = Image.fromarray(best_image_np)
    pil_image.save(f"{results_folder}/best_all.png")

    # Save the metrics
    results = pd.DataFrame({
        "iteration": list(range(1, NUM_ITERATIONS + 1)),
        "score": max_fit_list,
    })

    results.to_csv(f"{results_folder}/fitness_results.csv", index=False)

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
    for seed in seed_list:
        main(seed)
        print(f"Run with seed {seed} finished!")
