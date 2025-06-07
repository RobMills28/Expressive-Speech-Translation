# Docker/download_cosy_models_in_docker.py
from modelscope import snapshot_download
import os

# This script runs inside the Docker build, paths are relative to /app/CosyVoice
model_dir_base = 'pretrained_models' 
os.makedirs(model_dir_base, exist_ok=True)

models_to_download = {
    "CosyVoice2-0.5B": "iic/CosyVoice2-0.5B",
    "CosyVoice-ttsfrd": "iic/CosyVoice-ttsfrd"
}

for name, model_id in models_to_download.items():
    print(f"Downloading {name} ({model_id}) into ./{model_dir_base}/{name} ...")
    try:
        snapshot_download(model_id, local_dir=os.path.join(model_dir_base, name))
        print(f"Successfully downloaded {name}.")
    except Exception as e:
        print(f"ERROR downloading {name}: {e}")
        # Decide if you want the build to fail here or continue
        # For now, it will continue, but the service might not work if models are missing.

print("Model download process complete from within Docker build.")