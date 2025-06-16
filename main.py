import threading
import queue
from osc_sender import osc_sender
from video_encoder import Video_Encoder 

latent_queue = queue.Queue()
osc_thread = threading.Thread(target=osc_sender, args=(latent_queue,))
osc_thread.start()

encoder = Video_Encoder()
try:
    while True:
        latent_vec = encoder.image_to_latent()
        if latent_vec is not None:
            print(latent_vec)
        if latent_vec is not None:
            latent_queue.put(latent_vec)
except KeyboardInterrupt:
    print("Shutting down!")
    latent_queue.put(None) 
    encoder.close()
