import torch
from torch import autocast
from torchvision import transforms as tf
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import numpy as np

device = "cuda:1" if torch.cuda.is_available() else "cpu"

# Load the components individually
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)
tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder").to(device)
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet").to(device)
scheduler = PNDMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

# Move models to GPU if available
vae = vae.to(device)
unet = unet.to(device)
text_encoder = text_encoder.to(device)

# Tokenize the prompt
prompt = "emmy governance pessibrt corridor enable hostbritt seeking aishwarya unlawful hose disturbed reverscurrytimate my latvia gimmcyberpunk secretary opens emanuel sirikingjames warrancollins blk loaddecoration boxed drilling sively dq jamboburningcounsel 마sthelgofundme inccl oura coma revue anger simonwesley celebrates ious gurudev redhead flooding �roar redmond gmt raju falconvera prohibanglers moose lool acosta ominlt dering hoboken prodisisbrand ( emmys astro"
inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
input_ids = inputs.input_ids.to(device)

# Encode the prompt
with torch.no_grad():
    text_embeddings = text_encoder(input_ids)[0]

SEED = 42
generator = torch.Generator(device=device)
generator.manual_seed(SEED)

num_channels_latents = unet.in_channels
height = 512
width = 512
latents = torch.randn((1, num_channels_latents, height // 8, width // 8), device=device, generator=generator, requires_grad=False)

# Set scheduler parameters
scheduler.set_timesteps(50)  # Number of denoising steps
timesteps = scheduler.timesteps

# Denoising loop
for i, t in enumerate(timesteps):
    with autocast("cuda"):
        # Predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latents, t, encoder_hidden_states=text_embeddings).sample

    # Perform the actual denoising step
    latents = scheduler.step(noise_pred, t, latents).prev_sample

# Decode the latents to image
latents = 1 / 0.18215 * latents
with torch.no_grad():
    image = vae.decode(latents).sample

# Convert the image to a PIL image and save
image = (image / 2 + 0.5).clamp(0, 1)
image = image.cpu().permute(0, 2, 3, 1).numpy()
image = (image * 255).round().astype("uint8")
image = Image.fromarray(image[0])

image.save("test.png")
print("Image generated and saved as test.png")
