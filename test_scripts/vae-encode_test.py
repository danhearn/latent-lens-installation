import torch
from diffusers.models import AutoencoderKL
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
vae = vae.to(device).eval()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    #process img
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512)).astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5  
    img = np.transpose(img, (2, 0, 1))  
    input_tensor = torch.tensor(img).unsqueeze(0).to(device)

    with torch.no_grad():
        latent = vae.encode(input_tensor).latent_dist.mean  
        
    # Split each 64x64 feature map into 4 quadrants (32x32)
    q1 = latent[:, :, :32, :32]  # top-left
    q2 = latent[:, :, :32, 32:]  # top-right
    q3 = latent[:, :, 32:, :32]  # bottom-left
    q4 = latent[:, :, 32:, 32:]  # bottom-right

    # Global average pooling each quadrant
    p1 = q1.mean(dim=(2, 3))  # [1, 4]
    p2 = q2.mean(dim=(2, 3))  # [1, 4]
    p3 = q3.mean(dim=(2, 3))  # [1, 4]
    p4 = q4.mean(dim=(2, 3))  # [1, 4]

    # Concatenate into [1, 16]
    pooled_latent = torch.cat([p1, p2, p3, p4], dim=1)

    print("Latent vector:", pooled_latent.cpu().numpy().flatten())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()