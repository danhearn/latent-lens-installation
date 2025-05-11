import torch 
from diffusers.models import AutoencoderKL 

vae = AutoencoderKL.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder="vae")
vae.eval()

test_data = torch.randn((1, 3, 512, 512))

with torch.no_grad():
    latent = vae.encode(test_data).latent_dist.mean

print(latent.shape)