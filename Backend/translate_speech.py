from transformers import SeamlessM4TProcessor, SeamlessM4TModel
import torch
import torchaudio
import os
import numpy as np

# Load the processor and model
processor = SeamlessM4TProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4TModel.from_pretrained("facebook/seamless-m4t-v2-large")

def translate_audio(input_path, target_language="fra"):  # Changed function name to match import
    """
    Process and translate audio file
    """
    # Load your audio file
    audio, orig_freq = torchaudio.load(input_path)
    audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16000)

    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    # Convert to numpy array and flatten
    audio_numpy = audio.squeeze().numpy()

    # Process the audio input
    inputs = processor(
        audios=audio_numpy,
        sampling_rate=16000,
        src_lang="eng",  # Source language is English
        return_tensors="pt"
    )

    # Generate the translated speech
    with torch.no_grad():
        outputs = model.generate(**inputs, tgt_lang=target_language)

    # Process output and handle different possible formats
    if hasattr(outputs, 'waveform'):
        audio_output = outputs.waveform[0].cpu().numpy()
    elif isinstance(outputs, tuple) and len(outputs) > 0:
        audio_output = outputs[0].cpu().numpy()
    else:
        raise ValueError("Unexpected output format from model")

    # Convert the numpy array to a torch tensor and ensure it's 2D
    waveform_tensor = torch.tensor(audio_output).unsqueeze(0)

    # Save the output audio
    output_dir = os.path.dirname(input_path)
    output_path = os.path.join(output_dir, "translated_output.wav")
    torchaudio.save(output_path, waveform_tensor, sample_rate=16000)

    return output_path

if __name__ == "__main__":
    # Example usage / testing
    test_path = r"/Users/robmills/Documents/Audio Samples/English (US)/Arthur.mp3"
    output = translate_audio(test_path, "fra")
    print(f"Translation complete. Output saved as {output}")