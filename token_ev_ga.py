import torch
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, StableDiffusionPipeline
from deap import base, creator, tools, algorithms
import random
from PIL import Image
from io import BytesIO

NUM_GENERATIONS, CROSSOVER_PROB, MUTATION_PROB, IND_MUTATION_PROB = 10, 0.5, 0.2, 0.2

# Check if a GPU is available and if not, use the CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the components of the Stable Diffusion pipeline
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)
tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder").to(device)
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet").to(device)
scheduler = PNDMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

MIN_VALUE, MAX_VALUE = 0, tokenizer.vocab_size-3
START_OF_TEXT, END_OF_TEXT = tokenizer.bos_token_id, tokenizer.eos_token_id

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
    # Dummy function, replace with actual aesthetic evaluation logic
    # Here, we simply return a random score for demonstration
    return random.uniform(0, 1)

# Function to generate an image from text embeddings
def generate_image_from_embeddings(token_vector, num_inference_steps=25, guidance_scale=7.5):
    tmp_token_vector = np.insert(token_vector.cpu().detach().numpy().flatten(), 0, START_OF_TEXT)
    tmp_token_vector = np.append(tmp_token_vector, END_OF_TEXT)
    tmp_token_vector = torch.tensor(tmp_token_vector, dtype=torch.int64).to(device)
    tmp_token_vector = torch.clamp(tmp_token_vector, MIN_VALUE, MAX_VALUE).view(1, len(tmp_token_vector)).to(device)
    text_embeddings = text_encoder(tmp_token_vector)[0]
    height = 512
    width = 512
    num_channels_latents = unet.in_channels
    latents = torch.randn((1, num_channels_latents, height // 8, width // 8), device=device)
    scheduler.set_timesteps(num_inference_steps)
    for t in scheduler.timesteps:
        with torch.no_grad():
            noise_pred = unet(latents, t, encoder_hidden_states=text_embeddings)["sample"]
        latents = scheduler.step(noise_pred, t, latents)["prev_sample"]
    with torch.no_grad():
        image = vae.decode(latents / 0.18215).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    return image

# Tokenize and encode the initial prompt
prompt = ""
token_vector = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length - 2, return_tensors="pt").input_ids

# Genetic Algorithm setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, MIN_VALUE, MAX_VALUE)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, len(token_vector.flatten()))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    text_embeddings = torch.tensor(individual, device=device).unsqueeze(0)
    image = generate_image_from_embeddings(text_embeddings)
    score = aesthetic_evaluation(image)
    return score,

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=MIN_VALUE, up=MAX_VALUE, indpb=IND_MUTATION_PROB)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

def main():
    random.seed(42)
    population = toolbox.population(n=20)

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
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring

        fits = [ind.fitness.values[0] for ind in population]
        print(f"Gen {gen}: Max fitness {max(fits)}, Avg fitness {sum(fits) / len(fits)}")

    best_ind = tools.selBest(population, 1)[0]
    print("Best individual is %s, with fitness: %s" % (best_ind, best_ind.fitness.values))

    # Generate and display the best image
    best_text_embeddings = torch.tensor(best_ind, device=device).unsqueeze(0)
    best_image = generate_image_from_embeddings(best_text_embeddings)
    pil_image = Image.fromarray((best_image * 255).astype(np.uint8))
    pil_image.show()

if __name__ == "__main__":
    main()
