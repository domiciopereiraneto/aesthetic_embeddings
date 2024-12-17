import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
from deap import base, creator, tools, algorithms
import simulacra_rank_image
import os
import csv

# Load the pre-trained Stable Diffusion model
device = "cuda:1"
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to(device)

aesthetic_model = simulacra_rank_image.SimulacraAesthetic(device)

NUM_GENERATIONS, POP_SIZE, TOURNAMENT_SIZE = 5, 5, 3
CROSSOVER_PROB, MUTATION_PROB, IND_MUTATION_PROB = 0.8, 0.8, 0.2

MUTATION_MU, MUTATION_SIGMA = 0, 1
MIN_VALUE, MAX_VALUE = -3, 3

# Define the prompt for image generation
prompt = "A fantasy landscape with mountains, a river, and a castle"
num_inference_steps = 25
guidance_scale = 7.5
height = 512
width = 512

# Define the scheduler
pipe.scheduler.set_timesteps(num_inference_steps)

# Encode the prompt text
text_input = pipe.tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=77, truncation=True).to(device)
text_embeddings = pipe.text_encoder(text_input.input_ids.to(device))[0]

# Create unconditional embeddings for classifier-free guidance
uncond_input = pipe.tokenizer("", return_tensors="pt", padding="max_length", max_length=77, truncation=True).to(device)
uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0]

# Concatenate the unconditional and text embeddings
encoder_hidden_states = torch.cat([uncond_embeddings, text_embeddings])

# Define the aesthetic metric
def aesthetic_evaluation(image):
    aesthetic_score = aesthetic_model.predict(image)
    return aesthetic_score.item()

def evaluate(individual):
    image = generate_image_from_latents(individual)
    score = aesthetic_evaluation(image)
    return score,

# Define the fitness function
def generate_image_from_latents(individual):
    latents = torch.tensor(individual, device=device, dtype=torch.float32).view(1, pipe.unet.in_channels, height // 8, width // 8)
    latents = torch.clamp(latents, MIN_VALUE, MAX_VALUE)

    # Denoising loop
    for i, t in enumerate(pipe.scheduler.timesteps):
        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
        with torch.no_grad():
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states)["sample"]
        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = pipe.scheduler.step(noise_pred, t, latents)["prev_sample"]

    with torch.no_grad():
        image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor)["sample"]
    image = (image / 2 + 0.5).clamp(0, 1).squeeze()
    image = image.cpu().permute(1, 2, 0).numpy()
    image = (image * 255).round().astype("uint8")
    
    return image

if not os.path.exists("results"):
    os.mkdir("results")

# Set up DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.randn, pipe.unet.in_channels * (height // 8) * (width // 8))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=MUTATION_MU, sigma=MUTATION_SIGMA, indpb=IND_MUTATION_PROB)
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
toolbox.register("evaluate", evaluate)

# Initialize the population
population = toolbox.population(n=POP_SIZE)

# Create lists to store the statistics for each generation
mean_fitnesses = []
best_fitnesses = []
best_individuals = []

def record_stats(pop):
    fitnesses = [ind.fitness.values[0] for ind in pop]
    mean_fitness = np.mean(fitnesses)
    best_fitness = np.max(fitnesses)
    best_individual = tools.selBest(pop, 1)[0]

    mean_fitnesses.append(mean_fitness)
    best_fitnesses.append(best_fitness)
    best_individuals.append(best_individual)

# Run the evolutionary algorithm with stats recording
for gen in range(NUM_GENERATIONS):
    population = algorithms.varAnd(population, toolbox, cxpb=CROSSOVER_PROB, mutpb=MUTATION_PROB)
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    population = toolbox.select(population, len(population))
    record_stats(population)
    print(f"Generation {gen+1}: Mean Fitness = {mean_fitnesses[-1]}, Best Fitness = {best_fitnesses[-1]}")

    best_individual = tools.selBest(population, 1)[0]
    image = generate_image_from_latents(best_individual)
    image = Image.fromarray(image)
    image.save("results/best_"+str(gen+1)+".png")

# Save the statistics to a CSV file
with open("results/results.csv", "w", newline="") as csvfile:
    fieldnames = ["Generation", "Mean Fitness", "Best Fitness", "Best Individual"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for gen in range(NUM_GENERATIONS):
        writer.writerow({
            "Generation": gen + 1,
            "Mean Fitness": mean_fitnesses[gen],
            "Best Fitness": best_fitnesses[gen],
            "Best Individual": best_individuals[gen].tolist()
        })

# Get the best individual from the final population
best_ind = tools.selBest(population, 1)[0]

image = generate_image_from_latents(best_ind)
image = Image.fromarray(image)

# Save the generated image to disk
output_path = "results/best_all.png"
image.save(output_path)
print(f"Best image saved at {output_path}")
