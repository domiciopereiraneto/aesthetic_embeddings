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

import cma

# Load configuration from YAML file
config_path = "algorithms/config/config_hybrid.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Load parameters from the YAML file
SEED = config['seed']
SEED_PATH = config['seed_path']
cuda_n = str(config['cuda'])
predictor = config['predictor']
num_inference_steps = config['num_inference_steps']
guidance_scale = config['guidance_scale']
height = config['height']
width = config['width']
OUTPUT_FOLDER = config['output_folder']
NUM_GENERATIONS = config['num_generations']
POP_SIZE = config['pop_size']
SIGMA = config['sigma']
MAX_SCORE = config['max_score']
MIN_SCORE = config['min_score']
ADAM_ITERATIONS = config['adam_iterations']
model_id = config['model_id']

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

def evaluate(x, seed, initial_embedding):
    # x is a NumPy array representing the embedding vector
    # Convert it to a torch tensor
    with torch.no_grad():
        embedding = torch.tensor(x, dtype=torch.float32, device=device)
        # Reshape the embedding to the original shape
        embedding = embedding.view(initial_embedding.shape)
        image = generate_image_from_embeddings(embedding, seed)
        score = aesthetic_evaluation(image)[0].item()
    # CMA-ES minimizes the function, so we need to invert the score if higher is better

    return score  # Negate the fitness to turn maximization into minimization

def normalize_embedding_vector(x):
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std == 0:
        x_std = 1e-8  # Prevent division by zero
    x_normalized = (x - x_mean) / x_std
    return x_normalized

def adam_optimization(embedding, seed, n_iterations):
    text_embeddings = torch.nn.Parameter(embedding.clone())
    optimizer = torch.optim.Adam([text_embeddings], lr=1e-3, weight_decay=1e-5, eps=1e-4)  # Adjust learning rate as needed
    scaler = torch.cuda.amp.GradScaler()  # Initialize GradScaler for mixed precision training

    for _ in range(n_iterations):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(dtype=torch.float16):
            image = generate_image_from_embeddings(text_embeddings, seed)
            score = aesthetic_evaluation(image)
            loss = 1 / (1 + score)  # Negative because we want to maximize the score
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return text_embeddings.detach().cpu().numpy().flatten()

def main(seed, seed_number):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    results_folder = f"{OUTPUT_FOLDER}/results_{model_name}_{seed}"
    os.makedirs(results_folder, exist_ok=True)

    # Initialize the text embeddings with an empty prompt
    text_input = pipe.tokenizer(
        "",
        return_tensors="pt",
        padding="max_length",
        max_length=77,
        truncation=True
    ).to(device)
    text_embeddings_init = pipe.text_encoder(text_input.input_ids.to(device))[0]
    embedding_size = text_embeddings_init.numel()
    initial_embedding = text_embeddings_init.detach().cpu().numpy().flatten()    

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
    es = cma.CMAEvolutionStrategy(initial_embedding, SIGMA, es_options)

    with torch.no_grad():
        initial_image = generate_image_from_embeddings(text_embeddings_init, seed)
        image_np = initial_image.detach().clone().cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        pil_image.save(f"{results_folder}/it_0.png")

        initial_score = evaluate(initial_embedding, seed, text_embeddings_init)

    time_list = [0]
    best_fitness_overall = initial_score
    best_text_embeddings_overall = text_embeddings_init

    start_time = time.time()
    generation = 0

    max_score_list = [initial_score]
    avg_score_list = [initial_score]
    std_score_list = [0]

    while not es.stop():
        print(f"Generation {generation+1}/{NUM_GENERATIONS}")
        # Ask for new candidate solutions
        solutions = es.ask()
        # Optimize each solution using Adam
        optimized_solutions = [adam_optimization(torch.tensor(x, dtype=torch.float32, device=device).view(text_embeddings_init.shape), seed, ADAM_ITERATIONS) for x in solutions]
        # Evaluate optimized solutions
        tmp_fitnesses = []
        scores = []
        for x in optimized_solutions:
            score = evaluate(x, seed, text_embeddings_init)
            tmp_fitnesses.append(score)
            scores.append(score)
        # Tell CMA-ES the fitnesses
        es.tell(solutions, tmp_fitnesses)

        # Record statistics
        fitnesses = [-f for f in tmp_fitnesses]  # Convert back to positive scores

        max_score = max(scores)
        avg_score = np.mean(scores)
        std_score = np.std(scores)

        max_score_list.append(max_score)
        avg_score_list.append(avg_score)
        std_score_list.append(std_score)

        # Get best solution so far
        best_x = es.result.xbest
        best_fitness = -es.result.fbest  # Convert back to positive score

        with torch.no_grad():
            # Generate and save the best image
            best_text_embeddings = torch.tensor(best_x, dtype=torch.float32, device=device)
            best_text_embeddings = best_text_embeddings.view(text_embeddings_init.shape)
            best_image = generate_image_from_embeddings(best_text_embeddings, seed)
            image_np = best_image.detach().clone().cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)
            pil_image.save(results_folder + "/best_%d.png" % (generation+1))

        if best_fitness > best_fitness_overall:
            best_fitness_overall = best_fitness
            best_text_embeddings_overall = best_text_embeddings

        generation += 1

        elapsed_time = time.time() - start_time
        generations_done = generation
        generations_left = NUM_GENERATIONS - generations_done
        average_time_per_generation = elapsed_time / generations_done
        estimated_time_remaining = average_time_per_generation * generations_left

        formatted_time_remaining = format_time(estimated_time_remaining)

        time_list.append(elapsed_time)

        # Save the metrics
        results = pd.DataFrame({
            "generation": list(range(0, generation + 1)),
            "avg_score": avg_score_list,
            "std_score": std_score_list,
            "max_score": max_score_list,
            "elapsed_time": time_list
        })

        results.to_csv(f"{results_folder}/fitness_results.csv", index=False, na_rep='nan')

        # Plot and save the fitness evolution
        save_plot_results(results, results_folder)

        # Print stats
        print(f"Seed {seed_number} Generation {generation}/{NUM_GENERATIONS}: Max score: {max_score}, Avg score: {avg_score}, Estimated time remaining: {formatted_time_remaining}")

    # Save the overall best image
    with torch.no_grad():
        best_image = generate_image_from_embeddings(best_text_embeddings_overall, seed)
    best_image_np = best_image.detach().cpu().numpy()
    best_image_np = (best_image_np * 255).astype(np.uint8)
    pil_image = Image.fromarray(best_image_np)
    pil_image.save(f"{results_folder}/best_all.png")

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
    plot_mean_std(results['generation'], results['avg_score'], results['std_score'], "Population")
    plt.plot(results['generation'], results['max_score'], 'r-', label="Best")
    plt.ylim(0, 10)
    plt.xlabel('Generation')
    plt.ylabel('Aesthetic Score')
    plt.grid()
    plt.legend()
    plt.savefig(results_folder + "/score_evolution.png")

if __name__ == "__main__":
    i = 1
    for seed in seed_list:
        main(seed, i)
        print(f"Run with seed {seed} finished!")
        i += 1