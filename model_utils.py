import os
import requests

def download_model_if_missing():
    model_path = "models/retina_weights.weights.h5"
    if not os.path.exists(model_path):
        # Host your weights on a public link (e.g., GitHub Release, Dropbox)
        url = "YOUR_PUBLIC_DOWNLOAD_LINK_HERE"
        response = requests.get(url)
        with open(model_path, 'wb') as f:
            f.write(response.content)