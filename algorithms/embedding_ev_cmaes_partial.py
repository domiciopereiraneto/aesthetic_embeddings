import sys
import os
import json
import yaml

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to sys.path to obtain access to the submodules
sys.path.insert(0, parent_dir)

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from diffusers import StableDiffusionPipeline, DDIMScheduler
import random
from PIL import Image
import matplotlib.pyplot as plt
import time

# Aesthetic Evaluators
import nima_rank_image
import simulacra_rank_image
import laion_rank_image

import cma

config_path = "algorithms/config_cmaes_partial.yaml"

# Load configuration from YAML file
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
OUTPUT_FOLDER = config['output_folder']
NUM_GENERATIONS = config['num_generations']
POP_SIZE = config['pop_size']
SIGMA = config['sigma']
N_TOKENS = config['n_tokens']
MAX_SCORE = config['max_score']
MIN_SCORE = config['min_score']
FITNESS_WEIGHTS = config['fitness_weights']
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

def evaluate(input_embedding, seed, initial_embedding_shape, padding_embedding):
    # x is a NumPy array representing the embedding vector
    # Convert it to a torch tensor
    input_embedding_padded = np.concatenate([input_embedding, padding_embedding])
    with torch.no_grad():
        embedding = torch.tensor(input_embedding_padded, dtype=torch.float32, device=device)
        # Reshape the embedding to the original shape
        embedding = embedding.view(initial_embedding_shape)
        image = generate_image_from_embeddings(embedding, seed)
        score = aesthetic_evaluation(image)[0].item()
    # CMA-ES minimizes the function, so we need to invert the score if higher is better

    mean_embedding = torch.mean(embedding)
    std_embedding = torch.std(embedding)

    fitness = ( FITNESS_WEIGHTS[0]*(score - MIN_SCORE) / (MAX_SCORE - MIN_SCORE) +  FITNESS_WEIGHTS[1]*(1-abs(mean_embedding)) +  FITNESS_WEIGHTS[2]*(1-abs(std_embedding-1)) ).item()

    return -fitness, score  # Negate the score to turn maximization into minimization

def normalize_embedding_vector(x):
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std == 0:
        x_std = 1e-8  # Prevent division by zero
    x_normalized = (x - x_mean) / x_std
    return x_normalized

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
        max_length=N_TOKENS,
        truncation=True
    ).to(device)
    text_embeddings_init = pipe.text_encoder(text_input.input_ids.to(device))[0]
    embedding_size = text_embeddings_init.numel()
    initial_embedding = text_embeddings_init.detach().cpu().numpy().flatten()    

    padding_tokens = np.array([pipe.tokenizer.eos_token_id] * (77-N_TOKENS))
    padding_tokens = torch.tensor(padding_tokens, dtype=torch.int64).view(1, len(padding_tokens)).to(device)
    text_embeddings_padding = pipe.text_encoder(padding_tokens)[0]
    padding_embedding = text_embeddings_padding.detach().cpu().numpy().flatten()    

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

    text_embeddings_init_complete = torch.cat([text_embeddings_init, text_embeddings_padding], dim=1)
    text_embeddings_init_complete_shape = text_embeddings_init_complete.shape
    with torch.no_grad():
        initial_image = generate_image_from_embeddings(text_embeddings_init_complete, seed)
        image_np = initial_image.detach().clone().cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        pil_image.save(f"{results_folder}/it_0.png")

        initial_fitness, initial_score = evaluate(initial_embedding, seed, text_embeddings_init_complete_shape, padding_embedding)
        initial_fitness = initial_score

    time_list = [0]
    best_score_overall = initial_score
    best_fitness_overall = initial_fitness
    best_text_embeddings_overall = text_embeddings_init

    start_time = time.time()
    generation = 0

    max_fit_list = [initial_fitness]
    avg_fit_list = [initial_fitness]
    std_fit_list = [0]

    max_score_list = [initial_score]
    avg_score_list = [initial_score]
    std_score_list = [0]

    #embeddings_per_generation_list = []
    #fitnesses_per_generation_list = []

    best_embeddings_list = [initial_embedding]

    while not es.stop():
        print(f"Generation {generation+1}/{NUM_GENERATIONS}")

        os.makedirs(results_folder+"/gen_%d" % (generation+1), exist_ok=True)

        # Ask for new candidate solutions
        solutions = es.ask()
        solutions = [normalize_embedding_vector(x) for x in solutions]
        # Evaluate candidate solutions
        tmp_fitnesses = []
        scores = []
        for x in solutions:
            fitness, score = evaluate(x, seed, text_embeddings_init_complete_shape, padding_embedding)
            fitness = -score # remove for multiobjective optimization
            tmp_fitnesses.append(fitness)
            scores.append(score)
        # Tell CMA-ES the fitnesses
        es.tell(solutions, tmp_fitnesses)

        # Record statistics
        fitnesses = [-f for f in tmp_fitnesses]  # Convert back to positive scores

        max_fit = max(fitnesses)
        avg_fit = np.mean(fitnesses)
        std_fit = np.std(fitnesses)

        max_fit_list.append(max_fit)
        avg_fit_list.append(avg_fit)
        std_fit_list.append(std_fit)

        max_score = max(scores)
        avg_score = np.mean(scores)
        std_score = np.std(scores)

        max_score_list.append(max_score)
        avg_score_list.append(avg_score)
        std_score_list.append(std_score)

        #embeddings_per_generation_list.append(solutions)
        #fitnesses_per_generation_list.append(fitnesses)

        best_embeddings_list.append(es.result.xbest)

        # Get best solution so far
        best_x = es.result.xbest
        best_fitness = -es.result.fbest  # Convert back to positive score

        with torch.no_grad():
            ind_id = 1
            for solution in solutions:
                solution_tmp = torch.tensor(solution, dtype=torch.float32, device=device)
                solution_tmp = solution_tmp.view(text_embeddings_init.shape)
                solution_padded = torch.cat([solution_tmp, text_embeddings_padding], dim=1)
                image = generate_image_from_embeddings(solution_padded, seed)
                image_np = image.detach().clone().cpu().numpy()
                image_np = (image_np * 255).astype(np.uint8)
                pil_image = Image.fromarray(image_np)
                pil_image.save(results_folder + "/gen_%d/id_%d.png" % (generation+1, ind_id))
                ind_id += 1

        with torch.no_grad():
            # Generate and save the best image
            best_text_embeddings = torch.tensor(best_x, dtype=torch.float32, device=device)
            best_text_embeddings = best_text_embeddings.view(text_embeddings_init.shape)
            best_text_embeddings_padded = torch.cat([best_text_embeddings, text_embeddings_padding], dim=1)
            best_image = generate_image_from_embeddings(best_text_embeddings_padded, seed)
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
            "avg_fitness": avg_fit_list,
            "std_fitness": std_fit_list,
            "max_fitness": max_fit_list,
            "avg_score": avg_score_list,
            "std_score": std_score_list,
            "max_score": max_score_list,
            "elapsed_time": time_list
        })

        results.to_csv(f"{results_folder}/fitness_results.csv", index=False, na_rep='nan')

        # Convert embeddings to JSON strings
        embeddings_json = [json.dumps(embedding.tolist()) for embedding in solutions]

        results_generation = pd.DataFrame({
            "fitnesses": fitnesses,
            "embeddings": embeddings_json
        })

        results_generation.to_csv(f"{results_folder}/gen_{generation}/fitness_embeddings.csv", index=False, na_rep='nan')


        # Convert embeddings to JSON strings
        best_embeddings_json = [json.dumps(embedding.tolist()) for embedding in best_embeddings_list]
        results_best = pd.DataFrame({
            "generation": list(range(0, generation + 1)),
            "max_fitness": max_fit_list,
            "best_embeddings": best_embeddings_json
        })

        results_best.to_csv(f"{results_folder}/best_fitness_embeddings.csv", index=False, na_rep='nan')

        # Plot and save the fitness evolution
        save_plot_results(results, results_folder)

        # Print stats
        print(f"Seed {seed_number} Generation {generation}/{NUM_GENERATIONS}: Max fitness: {max_fit}, Avg fitness: {avg_fit}, Max score: {max_score}, Avg score: {avg_score}, Estimated time remaining: {formatted_time_remaining}")

    # Save the overall best image
    with torch.no_grad():
        best_text_embeddings_overall_padded = torch.cat([best_text_embeddings_overall, text_embeddings_padding], dim=1)
        best_image = generate_image_from_embeddings(best_text_embeddings_overall_padded, seed)
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
    plot_mean_std(results['generation'], results['avg_fitness'], results['std_fitness'], "Population")
    plt.plot(results['generation'], results['max_fitness'], 'r-', label="Best")
    plt.ylim(0, 3.5)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid()
    plt.legend()
    plt.savefig(results_folder + "/fitness_evolution.png")

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