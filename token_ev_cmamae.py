import torch
import torch.optim as optim
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler
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

BATCH_SIZE = 50  # Size of the population
NUM_ITERATIONS = 50
torch_device = "cuda:0"

aesthetic_model = simulacra_rank_image.SimulacraAesthetic(torch_device)

vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=True)
tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="text_encoder", use_safetensors=True
)
unet = UNet2DConditionModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="unet", use_safetensors=True
)

vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device)

prompt = [""]
height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion
num_inference_steps = 25  # Number of denoising steps
generator = torch.Generator(device=torch_device)
generator.manual_seed(24)
batch_size = len(prompt)

tokenized_input = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").input_ids

solution_dim = tokenized_input.numel()

archive = GridArchive(
    solution_dim=solution_dim,
    dims=(20, 20),
    ranges=[(0, tokenizer.vocab_size - 1), (0, (tokenizer.vocab_size - 1) * np.sqrt(solution_dim))],
    learning_rate=0.01,
    threshold_min=0.0
)

result_archive = GridArchive(solution_dim=solution_dim,
                             dims=(20, 20),
                             ranges=[(0, tokenizer.vocab_size - 1), (0, (tokenizer.vocab_size - 1) * np.sqrt(solution_dim))])

emitters = [
    EvolutionStrategyEmitter(
        archive,
        x0=np.random.randint(0, tokenizer.vocab_size, solution_dim),
        sigma0=10000,  # A larger initial sigma to explore a wider range
        ranker="imp",
        selection_rule="mu",
        restart_rule="basic",
        batch_size=BATCH_SIZE
    )
]

scheduler = Scheduler(archive, emitters, result_archive=result_archive)

# Randomly initialize the population of latent vectors
batch_latent_noise = torch.randn(
    (BATCH_SIZE, unet.config.in_channels, height // 8, width // 8),
    generator=generator,
    requires_grad=False,  # We don't need gradients for the population
    device=torch_device,
)

single_latent_noise = torch.randn(
    (1, unet.config.in_channels, height // 8, width // 8),
    generator=generator,
    requires_grad=False,  # We don't need gradients for the population
    device=torch_device,
)

if not os.path.exists("Outputs"):
    os.makedirs("Outputs")

def split_into_batches(tensor, split_batch_size):
    return [tensor[i:i + split_batch_size] for i in range(0, len(tensor), split_batch_size)]

def evaluate(individuals):
    solution_tensor = torch.tensor(individuals, dtype=torch.int64).to(torch_device)
    solution_tensor = torch.clamp(solution_tensor, 0, tokenizer.vocab_size - 1)
    tmp_latent = 1 / 0.18215 * denoising(solution_tensor)
    
    split_batch_size = 10
    # Split tmp_latent into batches
    latent_batches = split_into_batches(tmp_latent, split_batch_size)
    
    generated_images_list = []

    for latent_batch in latent_batches:
        with torch.no_grad():
            decoded_batch = vae.decode(latent_batch).sample
        
        decoded_batch = (decoded_batch / 2 + 0.5).clamp(0, 1).squeeze()
        decoded_batch = (decoded_batch.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()
        generated_images_list.append(decoded_batch)
    
    # Concatenate all the generated images
    generated_images = np.concatenate(generated_images_list, axis=0)

    objectives = []

    for image in generated_images:
        aesthetic_score = aesthetic_model.predict(image)
        objectives.append(aesthetic_score.item())

    return np.array(objectives)

def calculate_measures(individuals):
    measures = []
    for individual in individuals:
        individual_tmp = individual.astype(int)
        average_activation = np.mean(individual_tmp)
        weight_magnitude = np.linalg.norm(individual_tmp)
        measures.append([average_activation, weight_magnitude])
    return np.array(measures)

def denoising(individuals, batch=True):
    if not batch:
        tmp_latent = copy.deepcopy(single_latent_noise)
    else:
        tmp_latent = copy.deepcopy(batch_latent_noise)

    embeddings = text_encoder(individuals)[0]

    split_batch_size = 10
    latent_batches = split_into_batches(tmp_latent, split_batch_size)
    embedding_batches = split_into_batches(embeddings, split_batch_size)

    denoised_latents = []

    for latent_batch, embedding_batch in zip(latent_batches, embedding_batches):
        tmp_latent_batch = copy.deepcopy(latent_batch)

        embedding_batch = torch.cat([embedding_batch] * 2)

        ldm_scheduler = UniPCMultistepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
        ldm_scheduler.set_timesteps(num_inference_steps)
        for t in tqdm(ldm_scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([tmp_latent_batch] * 2)

            latent_model_input = ldm_scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=embedding_batch).sample

            # perform guidance
            noise_pred = noise_pred.chunk(2)[0]

            # compute the previous noisy sample x_t -> x_t-1
            tmp_latent_batch = ldm_scheduler.step(noise_pred, t, tmp_latent_batch).prev_sample

        denoised_latents.append(tmp_latent_batch)

    # Concatenate the results from all batches
    denoised_latents = torch.cat(denoised_latents, dim=0)

    return denoised_latents


def to_image(individual):
    solution_tensor = torch.tensor(individual, dtype=torch.int64)
    solution_tensor_2d = solution_tensor.view(1, tokenized_input.shape[1]).to(torch_device)
    solution_tensor_2d = torch.clamp(solution_tensor_2d, 0, tokenizer.vocab_size - 1)
    denoised_tensor = denoising(solution_tensor_2d, False)

    # Scale and decode the best latent vector with VAE
    tmp_latent = 1 / 0.18215 * denoised_tensor
    with torch.no_grad():
        tmp_img = vae.decode(tmp_latent).sample

    # Post-processing
    tmp_img = (tmp_img / 2 + 0.5).clamp(0, 1).squeeze()
    tmp_img = (tmp_img.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    tmp_img = Image.fromarray(tmp_img)

    return tmp_img

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
best_image = to_image(best_solution)
best_image.save("evolutionary_optimized_image.png")

solutions_df = scheduler.result_archive.data(["index","objective","measures","solution"], return_type="pandas")

solutions_df.to_csv("results.csv",index=False)

pd.DataFrame({"iteration": list(range(1,NUM_ITERATIONS+1)), "best_objective":best_objectives}).to_csv("results_iterations.csv",index=False)

i = 0

filename_list = []
objective_list = []
solution_list = []
prompt_list = []
for individual in best_solutions:
    tmp_img = to_image(individual['solution'])
    tmp_img.save("Outputs/it_"+str(i)+".png")

    tmp_solution = torch.tensor(individual['solution'], dtype=torch.int64)
    tmp_solution = torch.clamp(tmp_solution, 0, tokenizer.vocab_size - 1)
    decoded_string = tokenizer.decode(tmp_solution, skip_special_tokens=True, clean_up_tokenization_spaces = True)

    filename_list.append("it_"+str(i)+".png")
    objective_list.append(individual['objective'])
    prompt_list.append(decoded_string)
    solution_list.append(tmp_solution.numpy())
    i += 1

pd.DataFrame({"file_name": filename_list, "objective":objective_list, "prompt": prompt_list, "solution":solution_list}).to_csv("Outputs/results.csv",index=False)

iterations = list(range(1,NUM_ITERATIONS+1))

plt.figure()
plt.plot(iterations, objective_list)
plt.ylabel("Simulacra aesthetic score")
plt.xlabel("Iteration")
plt.savefig('results.png')