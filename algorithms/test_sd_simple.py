import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

device = "cuda:1"

# Load the pre-trained Stable Diffusion model
#model_id = "CompVis/stable-diffusion-v1-4"
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

# Define the prompt for image generation
prompt = "Beautiful sunset"
num_inference_steps = 12
guidance_scale = 7.5
height = 768
width = 768

# Encode the prompt text
text_input = pipe.tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=77, truncation=True).to(device)
text_embeddings = pipe.text_encoder(text_input.input_ids.to(device))[0]

# Create unconditional embeddings for classifier-free guidance
uncond_input = pipe.tokenizer("", return_tensors="pt", padding="max_length", max_length=77, truncation=True).to(device)
uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0]

# Concatenate the unconditional and text embeddings
encoder_hidden_states = torch.cat([uncond_embeddings, text_embeddings])

# Generate random latent vector or provide your own
# Here, we're generating a random latent vector. Replace this with your own latent vector if needed.
SEED = 42
generator = torch.Generator(device=device)
generator.manual_seed(SEED)
latents = torch.randn((1, pipe.unet.in_channels, height // 8, width // 8), generator=generator, device=device, requires_grad=False, dtype=torch.float16)

# Define the scheduler
pipe.scheduler.set_timesteps(num_inference_steps)

# Denoising loop
for i, t in enumerate(pipe.scheduler.timesteps):
    # Expand the latents if we are doing classifier-free guidance
    latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents

    # Predict the noise residual
    with torch.no_grad():
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states)["sample"]

    # Perform guidance
    if guidance_scale > 1.0:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # Compute the previous noisy sample x_t -> x_t-1
    latents = pipe.scheduler.step(noise_pred, t, latents)["prev_sample"]

# Decode the latent vector to an image
with torch.no_grad():
    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor)["sample"]

# Convert to PIL image
image = (image / 2 + 0.5).clamp(0, 1)
image = image.cpu().permute(0, 2, 3, 1).numpy()
image = (image * 255).round().astype("uint8")
image = Image.fromarray(image[0])

# Save the generated image to disk
os.makedirs("results/test", exist_ok=True)
output_path = "results/test/test.png"
image.save(output_path)

# Display the image
#image.show()

print(f"Image saved at {output_path}")
