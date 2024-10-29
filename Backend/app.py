import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import logging
import torch
import torchaudio
import tempfile
from pathlib import Path
from transformers import SeamlessM4TModel, SeamlessM4TProcessor
import numpy as np

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app, resources={
    r"/translate": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "supports_credentials": False
    }
})

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "facebook/seamless-m4t-v2-large"
auth_token = os.getenv('HUGGINGFACE_TOKEN')

if not auth_token:
    logger.error("HUGGINGFACE_TOKEN not found in environment variables")
    raise ValueError("Please set the HUGGINGFACE_TOKEN environment variable")

# Initialize model and processor
try:
    logger.info(f"Loading SeamlessM4T model: {MODEL_NAME}")
    processor = SeamlessM4TProcessor.from_pretrained(MODEL_NAME, token=auth_token)
    model = SeamlessM4TModel.from_pretrained(MODEL_NAME, token=auth_token)
    
    if torch.cuda.is_available():
        model = model.to('cuda')
        logger.info("Model loaded on GPU")
    else:
        logger.info("Model loaded on CPU")
    logger.info("Model and processor loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model or processor: {str(e)}", exc_info=True)
    model = None
    processor = None

# Language codes for SeamlessM4T
LANGUAGE_CODES = {
    'fra': 'fra',  # French
    'spa': 'spa',  # Spanish
    'deu': 'deu',  # German
    'ita': 'ita',  # Italian
    'por': 'por'   # Portuguese
}

# Mapping for short codes to full codes
LANGUAGE_MAP = {
    'de': 'deu',
    'fr': 'fra',
    'es': 'spa',
    'it': 'ita',
    'pt': 'por'
}

@app.route('/translate', methods=['POST', 'OPTIONS'])
def translate_audio():
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        logger.info("Received translation request")
        logger.info(f"Request form data: {request.form}")
        logger.info(f"Request files: {request.files}")
        
        # Validate file presence
        if 'file' not in request.files:
            logger.warning("No file part in the request")
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.warning("No selected file")
            return jsonify({'error': 'No selected file'}), 400
        
        # Validate target language
        target_language = request.form.get('target_language')
        logger.info(f"Received target language: {target_language}")
        
        if not target_language:
            logger.warning("No target language provided")
            return jsonify({'error': 'No target language provided'}), 400

        # Convert short code to model code if needed
        model_language = LANGUAGE_MAP.get(target_language, target_language)
        logger.info(f"Mapped language code: {model_language}")

        if model_language not in LANGUAGE_CODES:
            logger.warning(f"Unsupported target language: {target_language}")
            return jsonify({'error': f'Unsupported target language: {target_language}'}), 400
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_input, \
             tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_output:
            
            try:
                # Save uploaded file
                file.save(temp_input.name)
                logger.info(f"Temporary input file saved: {temp_input.name}")
                
                # Load and process the audio
                audio, orig_freq = torchaudio.load(temp_input.name)
                audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16000)
                
                # Convert to mono if stereo
                if audio.shape[0] > 1:
                    audio = torch.mean(audio, dim=0, keepdim=True)
                
                # Convert audio to numpy array and ensure it's in the correct format
                audio_numpy = audio.squeeze().numpy()
                
                # Process audio with the model processor
                inputs = processor(
                    audios=audio_numpy,  # Changed from audio to audios
                    sampling_rate=16000,
                    return_tensors="pt",
                    src_lang="eng",
                    tgt_lang=model_language,
                    task="s2st"  # Specify speech-to-speech translation task
                )
                
                # Move inputs to GPU if available
                if torch.cuda.is_available():
                    inputs = {name: tensor.to('cuda') for name, tensor in inputs.items()}
                
                # Generate translated speech
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        tgt_lang=model_language,
                        num_beams=5,
                        max_new_tokens=200,
                        use_cache=True
                    )
                    
                    # Get the waveform from the outputs
                    audio_output = outputs.waveform[0].cpu().numpy()
                
                # Save output audio
                torchaudio.save(
                    temp_output.name,
                    torch.tensor(audio_output).unsqueeze(0),
                    sample_rate=16000
                )
                
                logger.info(f"Translated audio saved: {temp_output.name}")
                return send_file(
                    temp_output.name,
                    mimetype='audio/wav',
                    as_attachment=True,
                    download_name=f"translated_{Path(file.filename).stem}.wav"
                )
                
            finally:
                # Clean up temporary files
                for temp_file in [temp_input.name, temp_output.name]:
                    try:
                        os.unlink(temp_file)
                        logger.info(f"Cleaned up temporary file: {temp_file}")
                    except Exception as e:
                        logger.error(f"Error cleaning up temporary file {temp_file}: {str(e)}")
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None and processor is not None,
        'gpu_available': torch.cuda.is_available()
    })

if __name__ == '__main__':
    if model is None or processor is None:
        logger.error("Could not start the app due to model loading failure.")
    else:
        logger.info("Starting Flask app on port 5001")
        app.run(debug=True, host='0.0.0.0', port=5001)