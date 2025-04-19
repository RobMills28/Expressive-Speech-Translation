# download_base_speakers.py
import os
import requests
import torch
from pathlib import Path

def download_file(url, destination):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    response = requests.get(url, stream=True)
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return destination

base_url = "https://huggingface.co/myshell-ai/OpenVoice/resolve/main/checkpoints_v2/base_speakers/ses/"
base_speakers = [
    "en-us.pth", "en-au.pth", "en-br.pth", "en-in.pth", "en-default.pth",
    "es.pth", "fr.pth", "zh.pth", "ja.pth", "ko.pth"
]

output_dir = "checkpoints_v2/base_speakers/ses"
os.makedirs(output_dir, exist_ok=True)

for speaker in base_speakers:
    url = base_url + speaker
    destination = os.path.join(output_dir, speaker)
    print(f"Downloading {speaker}...")
    download_file(url, destination)
    print(f"Saved to {destination}")

print("All base speaker embeddings downloaded successfully!")