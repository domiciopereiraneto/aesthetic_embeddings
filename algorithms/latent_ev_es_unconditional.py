import torch
import numpy as np
import pandas as pd
from diffusers import UNet2DModel, DDIMScheduler, VQModel
import tqdm
from deap import base, creator, tools
import random
from PIL import Image
import simulacra_rank_image
import laion_rank_image
import nima_rank_image
import copy
import argparse
import sys
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

CROSSOVER_PROB, MUTATION_PROB, IND_MUTATION_PROB = 0.7, 0.9, 0.2
NUM_GENERATIONS, POP_SIZE, TOURNMENT_SIZE, ELITISM = 10, 5, 3, 1  # Reduced for testing
LAMBDA = 0.1

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

num_inference_steps = 100
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
        aesthetic_score = aesthetic_model.predict(image)
    elif predictor == 1 or predictor == 2:
        pil_image = Image.fromarray(image)
        aesthetic_score = aesthetic_model.predict(pil_image)
    else:
        # Other metrics here
        return 0
    return aesthetic_score.item()

# Function to generate an image from noise vector
def generate_image_from_noise(noise):
    image = noise
    for t in tqdm.tqdm(scheduler.timesteps):
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

def evaluate(individual):
    # Convert the list back to a tensor
    noise = torch.tensor(individual, dtype=torch.float32, device=device)
    # Reshape the noise to the original shape
    noise = noise.view(1, unet.in_channels, unet.sample_size, unet.sample_size)
    image = generate_image_from_noise(noise)
    score = aesthetic_evaluation(image)
    return (score,)

def normalize_noise(individual):
    array = np.array(individual)
    mean = np.mean(array)
    std = np.std(array)
    if std == 0:
        std = 1e-8  # Prevent division by zero
    normalized = (array - mean) / std
    return creator.Individual(normalized)

def main(seed):
    generator = torch.manual_seed(seed)
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

    # Evolutionary Algorithm setup
    random.seed(seed)

    # DEAP setup
    if "FitnessMax" not in creator.__dict__:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMax)

    def initIndividual(icls, content):
        return icls(content + np.random.normal(0, 1, len(content)))
        #return icls(content)

    toolbox = base.Toolbox()
    toolbox.register("individual", initIndividual, creator.Individual, content=initial_noise)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=IND_MUTATION_PROB)
    toolbox.register("select", tools.selTournament, tournsize=TOURNMENT_SIZE)

    pop = toolbox.population(n=POP_SIZE)

    for ind in pop:
        ind = normalize_noise(ind)

    max_fit_list = []
    avg_fit_list = []
    std_fit_list = []
    best_list = []

    for gen in range(NUM_GENERATIONS):
        print(f"Generation {gen+1}/{NUM_GENERATIONS}")
        # Evaluate the individuals
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Record the best individual
        best_ind = tools.selBest(pop, 1)[0]
        max_fit_list.append(best_ind.fitness.values[0])
        avg_fit = np.mean([ind.fitness.values[0] for ind in pop])
        avg_fit_list.append(avg_fit)
        std_fit = np.std([ind.fitness.values[0] for ind in pop])
        std_fit_list.append(std_fit)
        best_list.append(copy.deepcopy(best_ind))

        # Print stats
        print("Generation %d: Max fitness: %f, Avg fitness: %f" % (gen+1, max_fit_list[-1], avg_fit_list[-1]))

        # Generate and save the best image
        best_noise = torch.tensor(best_ind, dtype=torch.float32, device=device)
        best_noise = best_noise.view(noise_shape)
        best_image = generate_image_from_noise(best_noise)
        best_image.save(results_folder + "/best_%d.png" % (gen+1))

        # Generate the next generation
        offspring = toolbox.select(pop, len(pop) - ELITISM)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CROSSOVER_PROB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTATION_PROB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # After applying crossover and mutation
        for ind in offspring:
            ind = normalize_noise(ind)

        # Add the elites
        elites = tools.selBest(pop, ELITISM)
        offspring.extend(elites)

        # Replace population
        pop[:] = offspring

    # Save the overall best image
    best_ind = tools.selBest(pop, 1)[0]
    best_noise = torch.tensor(best_ind, dtype=torch.float32, device=device)
    best_noise = best_noise.view(noise_shape)
    best_image = generate_image_from_noise(best_noise)
    best_image.save(results_folder + "/best_all.png")

    # Save the metrics
    results = pd.DataFrame({
        "generation": list(range(1, NUM_GENERATIONS + 1)),
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
