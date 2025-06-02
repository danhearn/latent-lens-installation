import torch
from diffusers.models import AutoencoderKL
import cv2
import numpy as np

class Video_Encoder: 
    def __init__(self): 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
        self.vae = self.vae.to(self.device).eval()
        self.cap = cv2.VideoCapture(0)
    
    def process_image(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        # process img
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512)).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5  # Normalize to [-1, 1]
        img = np.transpose(img, (2, 0, 1)) 
        return torch.tensor(img).unsqueeze(0).to(self.device)
    
    def encode_image(self, input_tensor): 
        with torch.no_grad():
            latent = self.vae.encode(input_tensor).latent_dist.mean
        return latent
    
    def pool_latent(self, latent):
        # Split each 64x64 feature map into 4 quadrants (32x32)
        q1 = latent[:, :, :32, :32]  # top-left
        q2 = latent[:, :, :32, 32:]  # top-right
        q3 = latent[:, :, 32:, :32]  # bottom-left
        q4 = latent[:, :, 32:, 32:]  # bottom-right

        # average pooling each quadrant
        p1 = q1.mean(dim=(2, 3))  # [1, 4]
        p2 = q2.mean(dim=(2, 3))  # [1, 4]
        p3 = q3.mean(dim=(2, 3))  # [1, 4]
        p4 = q4.mean(dim=(2, 3))  # [1, 4]

        pooled_latent = torch.cat([p1, p2, p3, p4], dim=1)
        normalised_latents = (pooled_latent - pooled_latent.min()) / (pooled_latent.max() - pooled_latent.min())
        
        return normalised_latents
    
    def image_to_latent(self):
        frame = self.process_image() 
        if frame is None:
            return None
        latent = self.encode_image(frame)
        pooled = self.pool_latent(latent) 
        return pooled.cpu().numpy().flatten()
    
    def close(self):
        if self.cap.isOpened():
            self.cap.release()
