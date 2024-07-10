import torch
import numpy as np
import pandas as pd
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, StableDiffusionPipeline
from PIL import Image
import simulacra_rank_image
from deap import base, creator, tools, algorithms
import random

# Check if a GPU is available and if not, use the CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the components of the Stable Diffusion pipeline
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)
tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder").to(device)
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet").to(device)
scheduler = PNDMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

aesthetic_model = simulacra_rank_image.SimulacraAesthetic(device)

NUM_GENERATIONS, POPULATION_SIZE = 10, 10
CROSSOVER_PROB, MUTATION_PROB, IND_MUTATION_PROB = 0.7, 0.2, 0.1

SEED = 1234
generator = torch.Generator(device=device)
generator.manual_seed(SEED)

num_channels_latents = unet.in_channels
height = 512
width = 512
latent_dim = (num_channels_latents, height // 8, width // 8)

# Fixed token vector
""" tmp_padding = [tokenizer.pad_token_id] * (tokenizer.model_max_length-2)
tmp_token_list = [tokenizer.bos_token_id]
tmp_token_list.extend(tmp_padding)
tmp_token_list.append(tokenizer.eos_token_id)
token_vector = torch.tensor(tmp_token_list, dtype=torch.int64).to(device)
tmp_token_vector = torch.tensor(token_vector, dtype=torch.int64).view(1, len(token_vector)).to(device)
text_embeddings = text_encoder(tmp_token_vector)[0] """
prompt = "cat"
inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=77, truncation=True).to(device)
text_embeddings = text_encoder(inputs.input_ids)[0]

pipeline = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=scheduler,
    safety_checker=None,
    feature_extractor=None,
)
pipeline = pipeline.to(device)

# Define the aesthetic evaluation function
def aesthetic_evaluation(image):
    aesthetic_score = aesthetic_model.predict(image)
    return aesthetic_score.item()

# Function to generate an image from text embeddings
def generate_image_from_latents(latents, num_inference_steps=25):
    scheduler.set_timesteps(num_inference_steps)
    tmp_latens = latents.clone().to(torch.float32)
    for t in scheduler.timesteps:
        with torch.no_grad():
            noise_pred = unet(tmp_latens, t, encoder_hidden_states=text_embeddings)["sample"]
        tmp_latens = scheduler.step(noise_pred, t, tmp_latens)["prev_sample"]
    with torch.no_grad():
        image = vae.decode(tmp_latens / 0.18215).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).round().astype("uint8")
    return image[0]

# Evolutionary Strategy setup with DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -1, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, np.prod(latent_dim))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    latent_vector = np.array(individual).reshape(latent_dim)
    latent_tensor = torch.tensor(latent_vector, device=device).view(1, *latent_dim)
    image = generate_image_from_latents(latent_tensor)
    score = aesthetic_evaluation(image)
    return score,

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=IND_MUTATION_PROB)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    random.seed(SEED)
    population = toolbox.population(n=POPULATION_SIZE)

    max_fit_list = []
    avg_fit_list = []

    for gen in range(NUM_GENERATIONS):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CROSSOVER_PROB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTATION_PROB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring

        fits = [ind.fitness.values[0] for ind in population]
        max_fit = max(fits)
        avg_fit = sum(fits) / len(fits)
        print(f"Gen {gen}: Max fitness {max_fit}, Avg fitness {avg_fit}")

        max_fit_list.append(max_fit)
        avg_fit_list.append(avg_fit)

        # Save the best image of the current generation
        best_ind = tools.selBest(population, 1)[0]
        best_latent_tensor = torch.tensor(np.array(best_ind).reshape(latent_dim), device=device).view(1, *latent_dim)
        best_image = generate_image_from_latents(best_latent_tensor)
        pil_image = Image.fromarray(best_image)
        pil_image.save(f"results/best_{gen+1}.png")

    # Save the final best image
    best_ind = tools.selBest(population, 1)[0]
    best_latent_tensor = torch.tensor(np.array(best_ind).reshape(latent_dim), device=device).view(1, *latent_dim)
    best_image = generate_image_from_latents(best_latent_tensor)
    pil_image = Image.fromarray(best_image)
    pil_image.save("results/best_all.png")

    # Save fitness results to a CSV file
    pd.DataFrame({"generation": list(range(1, NUM_GENERATIONS + 1)), "best_fitness": max_fit_list, "average_fitness": avg_fit_list}).to_csv("results/fitness_results.csv", index=False)

if __name__ == "__main__":
    main()
