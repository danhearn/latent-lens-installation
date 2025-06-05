import torch
from diffusers.models import AutoencoderKL
import cv2
import numpy as np
import NDIlib as ndi

class Video_Encoder: 
    def __init__(self): 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
        self.vae = self.vae.to(self.device).eval()
        self.cap = cv2.VideoCapture(0)

        if not ndi.initialize():
            print("NDI failed to init!!!!")
            sys.exit()
            return 0
        
        send_settings = ndi.SendCreate()
        send_settings.ndi_name = 'VAE_webcam'
        self.ndi_send = ndi.send_create(send_settings)
        self.ndi_frame = ndi.VideoFrameV2()
    
    def process_image(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        # send original image through ndi
        ndi_img = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        self.ndi_frame.data = ndi_img
        self.ndi_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRX
        ndi.send_send_video_v2(self.ndi_send, self.ndi_frame)

        vae_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # process img for VAE
        vae_img = cv2.resize(vae_img, (512, 512)).astype(np.float32) / 255.0
        img = (vae_img - 0.5) / 0.5  # Normalize to [-1, 1]
        vae_img = np.transpose(vae_img, (2, 0, 1)) 
        return torch.tensor(vae_img).unsqueeze(0).to(self.device)
    
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
        return pooled_latent
    
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
        ndi.send_destroy(self.ndi_send)
        ndi.destroy()