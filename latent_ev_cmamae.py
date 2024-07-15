import torch
import torch.optim as optim
from diffusers import StableDiffusionPipeline
import simulacra_rank_image
import numpy as np
import pandas as pd
from PIL import Image
import os
from tqdm.auto import tqdm
import copy
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler
import time
import matplotlib.pyplot as plt

BATCH_SIZE = 5  # Size of the population
NUM_ITERATIONS = 5
MIN_VALUE, MAX_VALUE = -4.753, 4.753
torch_device = "cuda:0"

aesthetic_model = simulacra_rank_image.SimulacraAesthetic(torch_device)

model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to(torch_device)

# Define the prompt for image generation
prompt = "A fantasy landscape with mountains, a river, and a castle"
num_inference_steps = 25
guidance_scale = 7.5
height = 512
width = 512

# Define the scheduler
pipe.scheduler.set_timesteps(num_inference_steps)

# Encode the prompt text
text_input = pipe.tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=77, truncation=True).to(torch_device)
text_embeddings = pipe.text_encoder(text_input.input_ids.to(torch_device))[0]

# Create unconditional embeddings for classifier-free guidance
uncond_input = pipe.tokenizer("", return_tensors="pt", padding="max_length", max_length=77, truncation=True).to(torch_device)
uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(torch_device))[0]

# Concatenate the unconditional and text embeddings
encoder_hidden_states = torch.cat([uncond_embeddings, text_embeddings])

solution_dim = pipe.unet.in_channels * (height // 8) * (width // 8)

archive = GridArchive(
    solution_dim=solution_dim,
    dims=(20, 20),
    ranges=[(MIN_VALUE, MAX_VALUE), (MIN_VALUE, MAX_VALUE)],
    learning_rate=0.01,
    threshold_min=0.0
)

result_archive = GridArchive(solution_dim=solution_dim,
                             dims=(20, 20),
                             ranges=[(MIN_VALUE, MAX_VALUE), (MIN_VALUE, MAX_VALUE)])

emitters = [
    EvolutionStrategyEmitter(
        archive,
        x0=np.random.randn(solution_dim),
        sigma0=0.1,  # A larger initial sigma to explore a wider range
        ranker="imp",
        selection_rule="mu",
        restart_rule="basic",
        batch_size=BATCH_SIZE
    )
]

scheduler = Scheduler(archive, emitters, result_archive=result_archive)

if not os.path.exists("Outputs"):
    os.makedirs("Outputs")

# Define the aesthetic metric
def aesthetic_evaluation(image):
    aesthetic_score = aesthetic_model.predict(image)
    return aesthetic_score.item()

def evaluate(individuals):
    objectives = []
    for tmp_individual in individuals:
        image = generate_image_from_latents(tmp_individual)
        score = aesthetic_evaluation(image)
        objectives.append(score)
    return np.array(objectives)

# Define the fitness function
def generate_image_from_latents(individual):
    latents = torch.tensor(individual, device=torch_device, dtype=torch.float32).view(1, pipe.unet.in_channels, height // 8, width // 8)
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

def calculate_measures(individuals):
    measures = []
    for individual in individuals:
        average_activation = np.mean(individual)
        #weight_magnitude = np.linalg.norm(individual)
        std_activation = np.std(individual)
        measures.append([average_activation, std_activation])
    return np.array(measures)

start_time = time.time()

best_objectives = []
best_solutions = []
for i in range(NUM_ITERATIONS):
    solutions = scheduler.ask()
    objectives = evaluate(solutions)
    measures = calculate_measures(solutions)
    scheduler.tell(objectives, measures)

    best_objective = scheduler.result_archive.best_elite['objective']
    best_objectives.append(best_objective)
    best_solutions.append(scheduler.result_archive.best_elite)

    print("Iteration "+str(i+1)+", best objective: "+str(best_objective))

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

best_solution = scheduler.result_archive.best_elite['solution']
best_image = generate_image_from_latents(best_solution)
best_image.save("results/best_all.png")

solutions_df = scheduler.result_archive.data(["index","objective","measures","solution"], return_type="pandas")

solutions_df.to_csv("results.csv",index=False)

pd.DataFrame({"iteration": list(range(1,NUM_ITERATIONS+1)), "best_objective":best_objectives}).to_csv("results_iterations.csv",index=False)

i = 0

filename_list = []
objective_list = []
solution_list = []
prompt_list = []
for individual in best_solutions:
    tmp_img = generate_image_from_latents(individual['solution'])
    tmp_img.save("results/best_"+str(i)+".png")

    filename_list.append("best_"+str(i)+".png")
    objective_list.append(individual)
    solution_list.append(individual)
    i += 1

pd.DataFrame({"file_name": filename_list, "objective":objective_list, "solution":solution_list}).to_csv("results/results.csv",index=False)

iterations = list(range(1,NUM_ITERATIONS+1))

plt.figure()
plt.plot(iterations, objective_list)
plt.ylabel("Simulacra aesthetic score")
plt.xlabel("Iteration")
plt.savefig('results/results.png')