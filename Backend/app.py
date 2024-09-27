from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import random
import logging

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simulate_translation(text, target_language):
    translations = {
        'es': 'Hola mundo, esto es una traducción simulada.',
        'fr': 'Bonjour le monde, ceci est une traduction simulée.',
        'de': 'Hallo Welt, dies ist eine simulierte Übersetzung.',
        'zh': '你好世界，这是一个模拟翻译。',
        'ja': 'こんにちは世界、これはシミュレーションされた翻译です。'
    }
    return translations.get(target_language, 'Hello world, this is a simulated translation.')

@app.route('/')
def home():
    return "LinguaSync AI Backend is running!"

@app.route('/translate', methods=['POST'])
def translate_audio():
    try:
        # Check if file is present in the request
        if 'file' not in request.files:
            logger.error("No file part in the request")
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        # Check if a file was selected
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({'error': 'No selected file'}), 400
        
        # Check file type (in a real app, you'd verify it's an audio file)
        if not file.filename.lower().endswith(('.mp3', '.wav', '.ogg', '.m4a')):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type. Please upload an audio file.'}), 400
        
        target_language = request.form.get('target_language')
        
        # Check if target language is provided
        if not target_language:
            logger.error("No target language provided")
            return jsonify({'error': 'Target language is required'}), 400
        
        # Simulate processing time
        time.sleep(2)
        
        # Generate a fake original text
        original_text = f"This is a simulated transcription of the uploaded audio file: {file.filename}"
        
        # Simulate translation
        translated_text = simulate_translation(original_text, target_language)
        
        logger.info(f"Successfully processed file: {file.filename}")
        
        return jsonify({
            'message': 'Translation complete',
            'original_text': original_text,
            'translated_text': translated_text,
            'output_file': f"simulated_output_{random.randint(1000, 9999)}.mp3"
        }), 200
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)