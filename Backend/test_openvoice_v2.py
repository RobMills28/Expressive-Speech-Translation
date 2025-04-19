# test_openvoice_v2.py
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

# Initialize converter
ckpt_converter = 'checkpoints_v2/converter'
device = "cpu"

# Load model
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

print("OpenVoice model loaded successfully!")
print("This test verifies the base model works without MeloTTS")