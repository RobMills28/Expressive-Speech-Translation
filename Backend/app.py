from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import logging
import torch
from transformers import AutoProcessor, SeamlessM4TModel
import librosa
import soundfile as sf
import tempfile
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "facebook/seamless-m4t-medium"
auth_token = os.getenv('HUGGINGFACE_TOKEN')

try:
    logger.info(f"Loading SeamlessM4T model: {MODEL_NAME}")
    model = SeamlessM4TModel.from_pretrained(MODEL_NAME, use_auth_token=auth_token)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, use_auth_token=auth_token)
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None
    processor = None

# Language code mapping
LANGUAGE_CODES = {
    'fr': 'fra',
    'es': 'spa',
    'de': 'deu',
    'zh': 'cmn',
    'ja': 'jpn'
}

def translate_audio(audio_file, target_language):
    # Load and preprocess the audio
    audio_array, sampling_rate = librosa.load(audio_file, sr=16000)
    inputs = processor(
        audio=audio_array,
        sampling_rate=sampling_rate,
        return_tensors="pt"
    )
    
    # Generate translation
    output_tokens = model.generate(**inputs, tgt_lang=LANGUAGE_CODES[target_language])
    
    # Convert tokens to audio
    output_audio = model.generate_speech(output_tokens, vocoder=model.vocoder)
    
    return output_audio.cpu().numpy()

@app.route('/translate', methods=['POST', 'OPTIONS'])
def translate_audio_route():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not file.filename.lower().endswith(('.mp3', '.wav', '.ogg', '.m4a')):
            return jsonify({'error': 'Invalid file type. Please upload an audio file.'}), 400
        
        target_language = request.form.get('target_language', 'fr')
        if target_language not in LANGUAGE_CODES:
            return jsonify({'error': f'Unsupported target language: {target_language}'}), 400
        
        # Save the uploaded file temporarily
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        file.save(temp_audio.name)
        
        # Translate the audio
        translated_audio = translate_audio(temp_audio.name, target_language)
        
        # Save the translated audio
        output_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(output_audio.name, translated_audio, 16000)
        
        return send_file(output_audio.name, as_attachment=True, download_name="translated_audio.wav")
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500
    
    finally:
        # Clean up temporary files
        if 'temp_audio' in locals():
            os.unlink(temp_audio.name)
        if 'output_audio' in locals():
            os.unlink(output_audio.name)

if __name__ == '__main__':
    if model is None or processor is None:
        logger.error("Could not start the app due to model loading failure.")
    else:
        app.run(debug=True, host='0.0.0.0', port=5000)