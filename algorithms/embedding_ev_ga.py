import sys
import os
import json

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
import argparse
import matplotlib.pyplot as plt
import time
from deap import base, creator, tools, algorithms

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

predictors = {0: 'SAM', 1: 'LAION', 2: 'NIMA'}

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
    cuda_n = str(2)
    print(f"CUDA device not provided, default is {cuda_n}")

if args.predictor is not None:
    predictor = args.predictor
else:
    predictor = 1 
    print(f"Aesthetic predictor not provided, default is {predictor} ({predictors[predictor]})")

num_inference_steps = 11
guidance_scale = 7.5
# Height and width of the images
height = 512
width = 512

OUTPUT_FOLDER = "results/ga_embedding_laion"
NUM_GENERATIONS, POP_SIZE = 100, 10  # Adjust as needed
CXPB, MUTPB, INDPB = 0.8, 0.2, 0.1  # Crossover and mutation probabilities
TOURNMENT_SIZE = 3
MUTATION_MU, MUTATION_SIGMA = 0, 0.2
MAX_SCORE, MIN_SCORE = 10.0, 1.0
FITNESS_WEIGHTS = [2.0, 1.0, 1.0]

# Check if a GPU is available and if not, use the CPU
device = "cuda:" + cuda_n if torch.cuda.is_available() else "cpu"

# Load the Stable Diffusion pipeline
model_id = "sd-legacy/stable-diffusion-v1-5"
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
    supports_grad = True
elif predictor == 1:
    # LAION Aesthetic Predictor
    aesthetic_model = laion_rank_image.LAIONAesthetic(device)
    model_name = 'LAION'
    supports_grad = True
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
        seed_list = [int(line.strip()) for line in file]

def generate_image_from_embeddings(text_embeddings, seed):
    uncond_input = pipe.tokenizer(
        "",
        return_tensors="pt",
        padding="max_length",
        max_length=77,
        truncation=True
    ).to(device)
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0]
    encoder_hidden_states = torch.cat([uncond_embeddings, text_embeddings], dim=0)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    latents = torch.randn(
        (1, pipe.unet.config.in_channels, height // 8, width // 8),
        generator=generator,
        device=device,
        dtype=torch.float32
    )
    for i, t in enumerate(pipe.scheduler.timesteps):
        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states)["sample"]
        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = pipe.scheduler.step(noise_pred, t, latents)["prev_sample"]
    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor)["sample"]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.squeeze(0).permute(1, 2, 0)
    return image

def aesthetic_evaluation(image):
    image_input = image.permute(2, 0, 1).to(torch.float32)
    if predictor == 0:
        score = aesthetic_model.predict_from_tensor(image_input)
    elif predictor == 1:
        score = aesthetic_model.predict_from_tensor(image_input)
    elif predictor == 2:
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

def evaluate(individual, seed, initial_embedding):
    with torch.no_grad():
        embedding = torch.tensor(individual, dtype=torch.float32, device=device)
        embedding = embedding.view(initial_embedding.shape)
        image = generate_image_from_embeddings(embedding, seed)
        score = aesthetic_evaluation(image)[0].item()
    mean_embedding = torch.mean(embedding)
    std_embedding = torch.std(embedding)
    fitness = (FITNESS_WEIGHTS[0] * (score - MIN_SCORE) / (MAX_SCORE - MIN_SCORE) +
               FITNESS_WEIGHTS[1] * (1 - abs(mean_embedding)) +
               FITNESS_WEIGHTS[2] * (1 - abs(std_embedding - 1))).item()
    #return -fitness,
    return (score,)

def normalize_embedding_vector(x):
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std == 0:
        x_std = 1e-8
    x_normalized = (x - x_mean) / x_std
    return x_normalized

def main(seed, seed_number):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    results_folder = f"{OUTPUT_FOLDER}/results_{model_name}_{seed}"
    os.makedirs(results_folder, exist_ok=True)

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

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.gauss)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=embedding_size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutGaussian, mu=MUTATION_MU, sigma=MUTATION_SIGMA, indpb=INDPB)
    toolbox.register("select", tools.selTournament, tournsize=TOURNMENT_SIZE)
    #toolbox.register("normalize", normalize_embedding_vector)
    toolbox.register("evaluate", evaluate, seed=seed, initial_embedding=text_embeddings_init)

    population = toolbox.population(n=POP_SIZE)

    os.makedirs(results_folder+"/gen_0", exist_ok=True)
    with torch.no_grad():
        ind_id = 1
        for solution in population:
            solution_tmp = torch.tensor(solution, dtype=torch.float32, device=device)
            solution_tmp = solution_tmp.view(text_embeddings_init.shape)
            image = generate_image_from_embeddings(solution_tmp, seed)
            image_np = image.detach().clone().cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)
            pil_image.save(results_folder + "/gen_0/id_%d.png" % (ind_id))
            ind_id += 1

    best_fitness_overall = 0
    best_text_embeddings_overall = None

    max_fit_list = []
    avg_fit_list = []
    std_fit_list = []
    time_list = []

    best_embeddings_list = []

    start_time = time.time()

    for gen in range(NUM_GENERATIONS):
        print(f"Generation {gen+1}/{NUM_GENERATIONS}")
        os.makedirs(results_folder+"/gen_%d" % (gen+1), exist_ok=True)

        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # for ind in offspring:
        #     toolbox.normalize(ind)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring

        with torch.no_grad():
            ind_id = 1
            for solution in population:
                solution_tmp = torch.tensor(solution, dtype=torch.float32, device=device)
                solution_tmp = solution_tmp.view(text_embeddings_init.shape)
                image = generate_image_from_embeddings(solution_tmp, seed)
                image_np = image.detach().clone().cpu().numpy()
                image_np = (image_np * 255).astype(np.uint8)
                pil_image = Image.fromarray(image_np)
                pil_image.save(results_folder + "/gen_%d/id_%d.png" % (gen+1, ind_id))
                ind_id += 1

        fits = [ind.fitness.values[0] for ind in population]
        max_fit = max(fits)
        avg_fit = sum(fits) / len(fits)
        std_fit = np.std(fits)

        max_fit_list.append(max_fit)
        avg_fit_list.append(avg_fit)
        std_fit_list.append(std_fit)

        elapsed_time = time.time() - start_time
        generations_done = gen + 1
        generations_left = NUM_GENERATIONS - generations_done
        average_time_per_generation = elapsed_time / generations_done
        estimated_time_remaining = average_time_per_generation * generations_left

        formatted_time_remaining = format_time(estimated_time_remaining)

        time_list.append(elapsed_time)

        best_ind = tools.selBest(population, 1)[0]
        best_embeddings_list.append(best_ind)
        with torch.no_grad():
            # Generate and save the best image
            best_text_embeddings = torch.tensor(best_ind, dtype=torch.float32, device=device)
            best_text_embeddings = best_text_embeddings.view(text_embeddings_init.shape)
            best_image = generate_image_from_embeddings(best_text_embeddings, seed)
            image_np = best_image.detach().clone().cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)
            pil_image.save(results_folder + "/best_%d.png" % (gen+1))

        if max_fit > best_fitness_overall:
            best_fitness_overall = max_fit
            best_text_embeddings_overall = best_text_embeddings

        # Save the metrics
        results = pd.DataFrame({
            "generation": list(range(1, gen + 2)),
            "avg_fitness": avg_fit_list,
            "std_fitness": std_fit_list,
            "max_fitness": max_fit_list,
            "elapsed_time": time_list
        })

        results.to_csv(f"{results_folder}/fitness_results.csv", index=False, na_rep='nan')

        # Convert embeddings to JSON strings
        embeddings_json = [json.dumps(np.array(embedding).tolist()) for embedding in population]

        results_generation = pd.DataFrame({
            "fitnesses": fits,
            "embeddings": embeddings_json
        })

        results_generation.to_csv(f"{results_folder}/gen_{gen+1}/fitness_embeddings.csv", index=False, na_rep='nan')

        # Convert embeddings to JSON strings
        best_embeddings_json = [json.dumps(np.array(embedding).tolist()) for embedding in best_embeddings_list]
        results_best = pd.DataFrame({
            "generation": list(range(1, gen + 2)),
            "max_fitness": max_fit_list,
            "best_embeddings": best_embeddings_json
        })

        results_best.to_csv(f"{results_folder}/best_fitness_embeddings.csv", index=False, na_rep='nan')

        # Plot and save the fitness evolution
        save_plot_results(results, results_folder)

        print(f"Seed {seed_number} Generation {gen+1}/{NUM_GENERATIONS}: Max fitness: {max_fit}, Avg fitness: {avg_fit}, Estimated time remaining: {formatted_time_remaining}")
    # text_input = pipe.tokenizer(
    #     "",
    #     return_tensors="pt",
    #     padding="max_length",
    #     max_length=77,
    #     truncation=True
    # ).to(device)
    # text_embeddings_init = pipe.text_encoder(text_input.input_ids.to(device))[0]
    # embedding_size = text_embeddings_init.numel()
    # initial_embedding = text_embeddings_init.detach().cpu().numpy().flatten()

    # creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    # creator.create("Individual", list, fitness=creator.FitnessMax)

    # toolbox = base.Toolbox()
    # toolbox.register("attr_float", random.uniform, -1, 1)
    # toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=embedding_size)
    # toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # toolbox.register("evaluate", evaluate, seed=seed, initial_embedding=initial_embedding)
    # toolbox.register("mate", tools.cxBlend, alpha=0.5)
    # toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    # toolbox.register("select", tools.selTournament, tournsize=3)

    # pop = toolbox.population(n=POP_SIZE)
    # hof = tools.HallOfFame(1)

    # stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("avg", np.mean)
    # stats.register("std", np.std)
    # stats.register("min", np.min)
    # stats.register("max", np.max)

    # algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NUM_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    # best_ind = hof[0]
    # best_embedding = torch.tensor(best_ind, dtype=torch.float32, device=device).view(text_embeddings_init.shape)
    # best_image = generate_image_from_embeddings(best_embedding, seed)
    # best_image_np = best_image.detach().cpu().numpy()
    # best_image_np = (best_image_np * 255).astype(np.uint8)
    # pil_image = Image.fromarray(best_image_np)
    # pil_image.save(f"{results_folder}/best_all.png")

    # results = pd.DataFrame(stats.compile(pop))
    # results.to_csv(f"{results_folder}/fitness_results.csv", index=False, na_rep='nan')

    # save_plot_results(results, results_folder)

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
    plt.ylim(0, 12)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid()
    plt.legend()
    plt.savefig(results_folder + "/fitness_evolution.png")

if __name__ == "__main__":
    i = 1
    for seed in seed_list:
        main(seed, i)
        print(f"Run with seed {seed} finished!")
        i += 1