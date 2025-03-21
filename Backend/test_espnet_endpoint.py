import os
import logging
import tempfile
from flask import Flask, request, jsonify
import torch
import torchaudio
import base64
import json

from services.espnet_backend import ESPnetBackend
from services.audio_processor import AudioProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a minimal Flask app for testing
app = Flask(__name__)

@app.route('/test-espnet', methods=['POST'])
def test_espnet_endpoint():
    """Test endpoint for ESPnet translation"""
    temp_files = []
    
    try:
        # Process the same way as your existing translation endpoint
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        target_language = request.form.get('target_language', 'fr')  # Default to French
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            file.save(temp_file.name)
            file_path = temp_file.name
            temp_files.append(file_path)
        
        # Process audio with your existing processor
        audio_processor = AudioProcessor()
        audio = audio_processor.process_audio(file_path)
        
        # Use the ESPnet backend
        espnet_backend = ESPnetBackend()
        result = espnet_backend.translate_speech(audio, target_lang=target_language)
        
        # Save the translated audio to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_output:
            temp_files.append(temp_output.name)
            
            waveform_tensor = result["audio"]
            
            torchaudio.save(
                temp_output.name,
                waveform_tensor,
                sample_rate=16000
            )
            
            with open(temp_output.name, 'rb') as audio_file:
                audio_data = audio_file.read()
            
            response_data = {
                'audio': base64.b64encode(audio_data).decode('utf-8'),
                'transcripts': result["transcripts"]
            }
            
            return jsonify(response_data)
            
    except Exception as e:
        logger.error(f"Error in test endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except Exception as e:
                logger.error(f"Error removing temp file {temp_file}: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True, port=5002)  # Use a different port than your main app