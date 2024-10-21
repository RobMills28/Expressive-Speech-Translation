import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import logging
import torch
from transformers import SeamlessM4TProcessor, SeamlessM4TForSpeechToSpeech
import torchaudio
import tempfile

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_NAME = "facebook/seamless-m4t-v2-large"
auth_token = os.getenv('HUGGINGFACE_TOKEN')

logger.info(f"Auth token: {'*' * len(auth_token) if auth_token else 'Not set'}")

try:
    logger.info(f"Loading SeamlessM4T model: {MODEL_NAME}")
    processor = SeamlessM4TProcessor.from_pretrained(MODEL_NAME, token=auth_token)
    model = SeamlessM4TForSpeechToSpeech.from_pretrained(MODEL_NAME, token=auth_token)
    logger.info("Model and processor loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model or processor: {str(e)}", exc_info=True)
    model = None
    processor = None

# Language code mapping
LANGUAGE_CODES = {
    'French': 'fra',
    'Spanish': 'spa',
    'German': 'deu',
    'Italian': 'ita',
    'Portuguese': 'por'
}

@app.route('/translate', methods=['POST', 'OPTIONS'])
def translate_audio_route():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        logger.info("Received translation request")
        if 'file' not in request.files:
            logger.warning("No file part in the request")
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.warning("No selected file")
            return jsonify({'error': 'No selected file'}), 400
        
        if not file.filename.lower().endswith(('.mp3', '.wav', '.ogg', '.m4a')):
            logger.warning(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type. Please upload an audio file.'}), 400
        
        target_language = request.form.get('target_language')
        if target_language not in LANGUAGE_CODES:
            logger.warning(f"Unsupported target language: {target_language}")
            return jsonify({'error': f'Unsupported target language: {target_language}'}), 400
        
        # Save the uploaded file temporarily
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        file.save(temp_input.name)
        logger.info(f"Temporary audio file saved: {temp_input.name}")
        
        # Load and process the audio
        audio, orig_freq = torchaudio.load(temp_input.name)
        audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16000)

        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        # Process the audio input
        inputs = processor(
            audios=audio.squeeze().numpy(),
            sampling_rate=16000,
            src_lang="eng",  # Assuming English as source language
            return_tensors="pt"
        )

        # Generate the translated speech
        with torch.no_grad():
            output = model.generate(**inputs, tgt_lang=LANGUAGE_CODES[target_language])

        # Extract the waveform from the output
        waveform = output[0].squeeze().cpu().numpy()

        # Save the output audio to a temporary file
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        torchaudio.save(temp_output.name, torch.tensor(waveform).unsqueeze(0), sample_rate=16000)
        logger.info(f"Translated audio saved: {temp_output.name}")

        return send_file(temp_output.name, as_attachment=True, download_name="translated_audio.wav")
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500
    
    finally:
        # Clean up temporary files
        if 'temp_input' in locals():
            os.unlink(temp_input.name)
            logger.info(f"Temporary input audio file deleted: {temp_input.name}")
        if 'temp_output' in locals():
            os.unlink(temp_output.name)
            logger.info(f"Temporary output audio file deleted: {temp_output.name}")

if __name__ == '__main__':
    if model is None or processor is None:
        logger.error("Could not start the app due to model loading failure.")
    else:
        logger.info("Starting Flask app")
        app.run(debug=True, host='0.0.0.0', port=5001)