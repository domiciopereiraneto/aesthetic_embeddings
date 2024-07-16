import torch
import numpy as np
import pandas as pd
from diffusers import StableDiffusionPipeline
from deap import base, creator, tools
import random
from PIL import Image
import simulacra_rank_image
import copy

CROSSOVER_PROB, MUTATION_PROB, IND_MUTATION_PROB = 0.7, 0.9, 0.2
NUM_GENERATIONS, POP_SIZE, TOURNMENT_SIZE, ELITISM = 50, 50, 3, 1
LAMBDA = 0.1

# Check if a GPU is available and if not, use the CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the components of the Stable Diffusion pipeline
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to(device)

num_inference_steps = 50
guidance_scale = 7.5

# Define the scheduler
pipe.scheduler.set_timesteps(num_inference_steps)

aesthetic_model = simulacra_rank_image.SimulacraAesthetic(device)

SEED = 54321
generator = torch.Generator(device=device)
generator.manual_seed(SEED)

num_channels_latents = pipe.unet.in_channels
height = 512
width = 512
latents = torch.randn((1, num_channels_latents, height // 8, width // 8), device=device, generator=generator, requires_grad=False)

MIN_VALUE, MAX_VALUE = 0, pipe.tokenizer.vocab_size-3
START_OF_TEXT, END_OF_TEXT = pipe.tokenizer.bos_token_id, pipe.tokenizer.eos_token_id
VECTOR_SIZE = 10

# Create unconditional embeddings for classifier-free guidance
uncond_input = pipe.tokenizer("", return_tensors="pt", padding="max_length", max_length=77, truncation=True).to(device)
uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0]

# Define the aesthetic evaluation function
def aesthetic_evaluation(image):
    aesthetic_score = aesthetic_model.predict(image)
    return aesthetic_score.item()

# Function to generate an image from text embeddings
def generate_image_from_embeddings(token_vector, num_inference_steps=25):
    tmp_token_vector = np.insert(token_vector.cpu().detach().numpy().flatten(), 0, START_OF_TEXT)

    padding_size =  pipe.tokenizer.model_max_length - len(tmp_token_vector.flatten())

    tmp_token_vector = np.append(tmp_token_vector, [END_OF_TEXT] * padding_size)
    tmp_token_vector = torch.tensor(tmp_token_vector, dtype=torch.int64).to(device)
    tmp_token_vector = torch.clamp(tmp_token_vector, MIN_VALUE, MAX_VALUE).view(1, len(tmp_token_vector)).to(device)

    text_embeddings = pipe.text_encoder(tmp_token_vector)[0]

    # Concatenate the unconditional and text embeddings
    encoder_hidden_states = torch.cat([uncond_embeddings, text_embeddings])

    tmp_latents = copy.deepcopy(latents)

    # Denoising loop
    for i, t in enumerate(pipe.scheduler.timesteps):
        latent_model_input = torch.cat([tmp_latents] * 2) if guidance_scale > 1.0 else tmp_latents
        with torch.no_grad():
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states)["sample"]
        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        tmp_latents = pipe.scheduler.step(noise_pred, t, tmp_latents)["prev_sample"]

    with torch.no_grad():
        image = pipe.vae.decode(tmp_latents / pipe.vae.config.scaling_factor)["sample"]
    image = (image / 2 + 0.5).clamp(0, 1).squeeze()
    image = image.cpu().permute(1, 2, 0).numpy()
    image = (image * 255).round().astype("uint8")
    
    return image

# Tokenize and encode the initial prompt
prompt = ""
#token_vector = pipe.tokenizer([""], padding="max_length", max_length=pipe.tokenizer.model_max_length - 2, return_tensors="pt").input_ids

# Genetic Algorithm setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, MIN_VALUE, MAX_VALUE)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, VECTOR_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    text_embeddings = torch.tensor(individual, device=device).unsqueeze(0)
    image = generate_image_from_embeddings(text_embeddings)
    score = aesthetic_evaluation(image)
    return score,

# Registering exponential mutation for integers
def mutExponential(individual, lambd, low, up, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            # Draw a number from an exponential distribution and add it to the gene
            delta = 1 + random.expovariate(lambd)
            # Randomly decide if the delta should be positive or negative
            if random.random() < 0.5:
                delta = -delta
            individual[i] += int(round(delta))
            # Ensure the mutated value is within bounds
            if individual[i] < low:
                individual[i] = low
            elif individual[i] > up:
                individual[i] = up
    return individual,

toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutUniformInt, low=MIN_VALUE, up=MAX_VALUE, indpb=IND_MUTATION_PROB)
#toolbox.register("mutate", mutExponential, lambd=LAMBDA, low=MIN_VALUE, up=MAX_VALUE, indpb=IND_MUTATION_PROB)
toolbox.register("select", tools.selTournament, tournsize=TOURNMENT_SIZE)
toolbox.register("evaluate", evaluate)

def main():
    random.seed(42)
    population = toolbox.population(n=POP_SIZE)

    max_fit_list = []
    avg_fit_list = []
    best_list = []
    prompt_list = []
    for gen in range(NUM_GENERATIONS):
        elites = tools.selBest(population, ELITISM)
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

        offspring = offspring + elites
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring

        fits = [ind.fitness.values[0] for ind in population]
        max_fit = max(fits)
        avg_fit = sum(fits) / len(fits)
        print(f"Gen {gen}: Max fitness {max_fit}, Avg fitness {avg_fit}")

        # Generate and display the best image
        best_ind = tools.selBest(population, 1)[0]
        prompt = detokenize(best_ind)

        max_fit_list.append(max_fit)
        avg_fit_list.append(avg_fit)
        best_list.append(best_ind)
        prompt_list.append(prompt)

        best_text_embeddings = torch.tensor(best_ind, device=device).unsqueeze(0)
        best_image = generate_image_from_embeddings(best_text_embeddings)
        pil_image = Image.fromarray((best_image))
        pil_image.save("results/best_%d.png" % (gen+1))

    best_ind = tools.selBest(population, 1)[0]
    print("Best individual is %s, with fitness: %s" % (best_ind, best_ind.fitness.values))

    # Generate and display the best image
    best_text_embeddings = torch.tensor(best_ind, device=device).unsqueeze(0)
    best_image = generate_image_from_embeddings(best_text_embeddings)
    pil_image = Image.fromarray((best_image))
    pil_image.save("results/best_all.png")

    pd.DataFrame({"generation": list(range(1,NUM_GENERATIONS+1)), "best_fitness": max_fit_list, "average_fitness": avg_fit_list,
                  "best_individual": best_list, "prompt": prompt_list}).to_csv("results/fitness_results.csv", index=False)

def detokenize(individual):
    tmp_solution = torch.tensor(individual, dtype=torch.int64)
    tmp_solution = torch.clamp(tmp_solution, 0, pipe.tokenizer.vocab_size - 1)
    decoded_string = pipe.tokenizer.decode(tmp_solution, skip_special_tokens=True, clean_up_tokenization_spaces = True)

    return decoded_string

if __name__ == "__main__":
    main()
