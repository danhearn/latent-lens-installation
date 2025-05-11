import queue
from pythonosc.udp_client import SimpleUDPClient

def osc_sender(latent_queue, ip="127.0.0.1", pd_port=9998, touch_port=9999):
    pd_client = SimpleUDPClient(ip, pd_port)
    touch_client = SimpleUDPClient(ip, touch_port)
    while True:
        try:
            latent = latent_queue.get(timeout=1)
            if latent is None:
                break
            
            pd_client.send_message("/latent", latent.tolist())
            touch_client.send_message("/latent", latent.tolist())
        except queue.Empty:
            continue
