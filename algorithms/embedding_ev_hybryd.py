import sys
import os
import argparse
import numpy as np
import pandas as pd
import random
import time
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image
from torch.cuda.amp import autocast, GradScaler
import cma

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to sys.path to obtain access to the submodules
sys.path.insert(0, parent_dir)

# Aesthetic Evaluators
import nima_rank_image
import simulacra_rank_image
import laion_rank_image

# Argument parser
parser = argparse.ArgumentParser(description='Combined optimization using Adam and CMA-ES.')

parser.add_argument('--seed', type=int, help='Seed')
parser.add_argument('--seed_path', type=str, help='Path to seed list file')
parser.add_argument('--cuda', type=int, help='CUDA GPU to use')
parser.add_argument('--predictor', type=int, help='Aesthetic predictor to use\n0 - SAM\n1 - LAION\n2 - NIMA')
parser.add_argument('--adam_steps', type=int, default=5, help='Number of Adam optimization steps per generation')

args = parser.parse_args()

# Seed and device setup
if args.seed_path is not None:
    SEED_PATH = args.seed_path
elif args.seed is not None:
    SEED = args.seed
    SEED_PATH = None
else:
    SEED = 42
    SEED_PATH = None

cuda_n = str(args.cuda) if args.cuda is not None else str(0)
predictor = args.predictor if args.predictor is not None else 0

# Image generation parameters
num_inference_steps = 11
guidance_scale = 7.5
height = 512
width = 512
NUM_GENERATIONS, POP_SIZE = 200, 10
SIGMA = 0.2

# Check if a GPU is available
device = "cuda:" + cuda_n if torch.cuda.is_available() else "cpu"

# Load the Stable Diffusion pipeline
model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
pipe.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe.to(device)
pipe.scheduler.set_timesteps(num_inference_steps)

# Initialize the aesthetic model
if predictor == 0:
    aesthetic_model = simulacra_rank_image.SimulacraAesthetic(device)
elif predictor == 1:
    aesthetic_model = laion_rank_image.LAIONAesthetic(device)
elif predictor == 2:
    aesthetic_model = nima_rank_image.NIMAAesthetics(device)
else:
    raise ValueError("Invalid predictor option.")

# Load seed list
if SEED_PATH is None:
    seed_list = [SEED]
else:
    with open(SEED_PATH, 'r') as file:
        seed_list = [int(line.strip()) for line in file]

def generate_image_from_embeddings(text_embeddings, seed):
    uncond_input = pipe.tokenizer("", return_tensors="pt", padding="max_length", max_length=77, truncation=True).to(device)
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0]
    encoder_hidden_states = torch.cat([uncond_embeddings, text_embeddings], dim=0)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    latents = torch.randn((1, pipe.unet.config.in_channels, height // 8, width // 8), generator=generator, device=device)

    for t in pipe.scheduler.timesteps:
        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states)["sample"]

        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = pipe.scheduler.step(noise_pred, t, latents)["prev_sample"]

    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor)["sample"]
    return (image / 2 + 0.5).clamp(0, 1).squeeze(0).permute(1, 2, 0)

def aesthetic_evaluation(image):
    image_input = image.permute(2, 0, 1).to(torch.float32)
    return aesthetic_model.predict_from_tensor(image_input)

def evaluate(x, seed, initial_embedding):
    with torch.no_grad():
        embedding = torch.tensor(x, dtype=torch.float32, device=device).view(initial_embedding.shape)
        image = generate_image_from_embeddings(embedding, seed)
        score = aesthetic_evaluation(image)[0].item()
    return -score

def main(seed, seed_number):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    results_folder = f"results/test/results_combined_{seed}"
    os.makedirs(results_folder, exist_ok=True)

    text_input = pipe.tokenizer("", return_tensors="pt", padding="max_length", max_length=77, truncation=True).to(device)
    text_embeddings_init = pipe.text_encoder(text_input.input_ids.to(device))[0]
    embedding_size = text_embeddings_init.numel()
    initial_embedding = text_embeddings_init.detach().cpu().numpy().flatten()

    es = cma.CMAEvolutionStrategy(initial_embedding, SIGMA, {'popsize': POP_SIZE, 'maxiter': NUM_GENERATIONS})

    best_score_overall = aesthetic_evaluation(generate_image_from_embeddings(text_embeddings_init, seed))[0].item()
    best_text_embeddings_overall = text_embeddings_init

    for generation in range(NUM_GENERATIONS):
        print(f"Generation {generation + 1}/{NUM_GENERATIONS}")
        solutions = es.ask()

        for _ in range(args.adam_steps):
            for x in solutions:
                embedding = torch.tensor(x, dtype=torch.float32, device=device).view(text_embeddings_init.shape)
                optimizer = torch.optim.Adam([embedding], lr=1e-3)
                optimizer.zero_grad()
                image = generate_image_from_embeddings(embedding, seed)
                score = aesthetic_evaluation(image)
                loss = -score
                loss.backward()
                optimizer.step()

        fitnesses = [evaluate(x, seed, text_embeddings_init) for x in solutions]
        es.tell(solutions, fitnesses)

        best_x = es.result.xbest
        best_score = -es.result.fbest

        if best_score > best_score_overall:
            best_score_overall = best_score
            best_text_embeddings_overall = torch.tensor(best_x, dtype=torch.float32, device=device).view(text_embeddings_init.shape)

    with torch.no_grad():
        best_image = generate_image_from_embeddings(best_text_embeddings_overall, seed)
    best_image_np = (best_image.detach().cpu().numpy() * 255).astype(np.uint8)
    pil_image = Image.fromarray(best_image_np)
    pil_image.save(f"{results_folder}/best_all.png")

if __name__ == "__main__":
    for i, seed in enumerate(seed_list, start=1):
        main(seed, i)
        print(f"Run with seed {seed} finished!")