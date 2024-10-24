import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import logging
import torch
from transformers import SeamlessM4TProcessor, SeamlessM4TForSpeechToSpeech
import torchaudio
import tempfile
from pathlib import Path

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

logger.info(f"Auth token: {'*' * len(auth_token) if auth_token else 'Not set'}")

# Initialize model and processor
try:
    logger.info(f"Loading SeamlessM4T model: {MODEL_NAME}")
    processor = SeamlessM4TProcessor.from_pretrained(MODEL_NAME, token=auth_token)
    model = SeamlessM4TForSpeechToSpeech.from_pretrained(MODEL_NAME, token=auth_token)
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

# Language codes for SeamlessM4T - direct mapping for frontend values
LANGUAGE_CODES = {
    'fra': 'fra',
    'spa': 'spa',
    'deu': 'deu',
    'ita': 'ita',
    'por': 'por'
}

@app.route('/translate', methods=['POST', 'OPTIONS'])
def translate_audio():
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        logger.info("Received translation request")
        logger.info(f"Request form data: {request.form}")
        logger.info(f"Request files: {request.files}")
        logger.info(f"Received target language: {request.form.get('target_language')}")
        logger.info(f"Available languages: {list(LANGUAGE_CODES.keys())}")
        
        # Validate file presence
        if 'file' not in request.files:
            logger.warning("No file part in the request")
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.warning("No selected file")
            return jsonify({'error': 'No selected file'}), 400
        
        # Validate file type
        allowed_extensions = {'.mp3', '.wav', '.ogg', '.m4a'}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            logger.warning(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type. Please upload an audio file (MP3, WAV, OGG, or M4A)'}), 400
        
        # Validate target language
        target_language = request.form.get('target_language')
        if not target_language or target_language not in LANGUAGE_CODES:
            logger.warning(f"Unsupported target language: {target_language}")
            return jsonify({'error': f'Unsupported target language: {target_language}'}), 400
        # Add this right after getting target_language
        target_language = request.form.get('target_language')
        logger.info(f"Received target language: '{target_language}'")
        logger.info(f"Available language codes: {list(LANGUAGE_CODES.keys())}")
        
        # Create temporary files
        temp_input = tempfile.NamedTemporaryFile(suffix=file_ext, delete=False)
        temp_output = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        
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
            
            # Process audio input
            inputs = processor(
                audios=audio.squeeze().numpy(),
                sampling_rate=16000,
                src_lang="eng",  # Source language is English
                return_tensors="pt"
            )
            
            # Move inputs to GPU if available
            if torch.cuda.is_available():
                inputs = {name: tensor.to('cuda') for name, tensor in inputs.items()}
            
            # Generate translated speech
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    tgt_lang=LANGUAGE_CODES[target_language]
                )
            
            # Move output back to CPU if necessary
            if torch.cuda.is_available():
                output = output.cpu()
            
            # Extract waveform and save
            waveform = output[0].squeeze().numpy()
            torchaudio.save(
                temp_output.name,
                torch.tensor(waveform).unsqueeze(0),
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
            try:
                os.unlink(temp_input.name)
                os.unlink(temp_output.name)
                logger.info("Temporary files cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up temporary files: {str(e)}")
    
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