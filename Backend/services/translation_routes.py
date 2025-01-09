from flask import request, jsonify, Response
import hashlib
import time
import torch
import numpy as np
import base64
import json
import tempfile
from pathlib import Path
import logging
import torchaudio
from pydub import AudioSegment

from .audio_processor import AudioProcessor
from .resource_monitor import ResourceMonitor
from .error_handler import ErrorHandler
from .utils import cleanup_file

logger = logging.getLogger(__name__)

def handle_translation(processor, model, text_model, tokenizer, DEVICE, SAMPLE_RATE, LANGUAGE_MAP, LANGUAGE_CODES):
    request_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
    start_time = time.time()
    temp_files = []
    source_text = "Text extraction unavailable"
    target_text = "Text extraction unavailable"
    
    try:
        logger.info(f"Starting translation request {request_id}")
        
        # Resource check
        resources_ok, resource_error = ResourceMonitor.check_resources()
        if not resources_ok:
            raise ValueError(f"Resource check failed: {resource_error}")
        
        # Validate request
        if 'file' not in request.files:
            logger.error(f"Request {request_id}: No file in request")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if not file.filename:
            logger.error(f"Request {request_id}: Empty filename")
            return jsonify({'error': 'No selected file'}), 400
        
        target_language = request.form.get('target_language')
        if not target_language:
            logger.error(f"Request {request_id}: No target language specified")
            return jsonify({'error': 'No target language specified'}), 400

        # Validate language
        model_language = LANGUAGE_MAP.get(target_language, target_language)
        if model_language not in LANGUAGE_CODES:
            logger.error(f"Request {request_id}: Unsupported language {target_language}")
            return jsonify({'error': f'Unsupported language: {target_language}'}), 400

        # Initialize audio processor
        try:
            audio_processor = AudioProcessor()
            if not audio_processor.diagnostics:
                logger.warning(f"Request {request_id}: AudioDiagnostics not initialized properly")
        except Exception as e:
            logger.error(f"Failed to initialize AudioProcessor: {str(e)}")
            return jsonify({'error': 'Internal processing error'}), 500

        # Process audio file
        file_extension = Path(file.filename).suffix.lower()
        try:
            if file_extension == '.mp3':
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as wav_file:
                    temp_files.append(wav_file.name)
                    file.save(wav_file.name + '.mp3')
                    temp_files.append(wav_file.name + '.mp3')
                    sound = AudioSegment.from_mp3(wav_file.name + '.mp3')
                    sound.export(wav_file.name, format='wav')
                    audio_path = wav_file.name
            else:
                with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_input:
                    temp_files.append(temp_input.name)
                    file.save(temp_input.name)
                    audio_path = temp_input.name

            if not Path(audio_path).exists() or Path(audio_path).stat().st_size == 0:
                raise ValueError("Failed to save audio file")
            
            logger.info(f"Request {request_id}: Saved input file: {audio_path}")
        except Exception as e:
            logger.error(f"File handling error: {str(e)}")
            return jsonify({'error': 'Failed to process uploaded file'}), 400

        # Validate audio length
        is_valid, error_message = audio_processor.validate_audio_length(audio_path)
        if not is_valid:
            logger.error(f"Request {request_id}: Audio validation failed - {error_message}")
            return jsonify({'error': error_message}), 400

        # Process audio
        try:
            logger.info(f"Request {request_id}: Beginning audio processing with diagnostics")
            
            audio, input_diagnostics = audio_processor.process_audio_enhanced(
                audio_path,
                target_language=model_language,
                return_diagnostics=True
            )
            
            if input_diagnostics:
                input_report = audio_processor.diagnostics.generate_report(input_diagnostics, model_language)
                if input_report:
                    logger.info("\n" + "="*50)
                    logger.info("Input Audio Quality Report")
                    logger.info("="*50)
                    logger.info(f"Request ID: {request_id}")
                    logger.info(f"Target Language: {model_language}")
                    logger.info("-"*50)
                    logger.info(input_report)
                    logger.info("="*50 + "\n")
                
                if 'metrics' in input_diagnostics:
                    for metric, value in input_diagnostics['metrics'].items():
                        logger.debug(f"Input Quality Metric - {metric}: {value}")
            
            audio_numpy = audio.squeeze().numpy()
            
            if np.isnan(audio_numpy).any() or np.isinf(audio_numpy).any():
                raise ValueError("Invalid audio data detected")
                
            logger.info(f"Request {request_id}: Audio processed successfully")
            
        except Exception as e:
            logger.error(f"Audio processing error: {str(e)}")
            return jsonify({'error': f'Failed to process audio: {str(e)}'}), 400
        
        # Model processing
        try:
            inputs = processor(
                audios=audio_numpy,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                src_lang="eng",
                tgt_lang=model_language,
                padding=True,
                truncation=True,
                max_length=256000
            )
            
            logger.info(f"Input keys: {inputs.keys()}")
            
            if not inputs or not inputs.keys():
                raise ValueError("Failed to prepare model inputs")
            
            if DEVICE.type == 'cuda':
                inputs = {name: tensor.to(DEVICE) for name, tensor in inputs.items()}

            # Generate source text
            try:
                with torch.no_grad():
                    source_outputs = text_model.generate(
                        input_features=inputs["input_features"],
                        tgt_lang="eng",
                        num_beams=8,
                        do_sample=False,
                        max_new_tokens=1000,
                        temperature=0.2,
                        length_penalty=1.0,
                        repetition_penalty=1.5,
                        no_repeat_ngram_size=3
                    )
                    source_text = processor.batch_decode(source_outputs, skip_special_tokens=True)[0]
                    
                    if any(phrase in source_text for phrase in [" H. H. H", ", the, the", " of the, of the"]):
                        logger.info("Detected potential repetition, performing second pass")
                        source_outputs = text_model.generate(
                            input_features=inputs["input_features"],
                            tgt_lang="eng",
                            num_beams=8,
                            do_sample=False,
                            max_new_tokens=1000,
                            repetition_penalty=2.0,
                            no_repeat_ngram_size=4,
                            temperature=0.2
                        )
                        source_text = processor.batch_decode(source_outputs, skip_special_tokens=True)[0]
                    
                    logger.info(f"Source text: {source_text}")
                    
            except Exception as e:
                logger.error(f"Source text generation error: {str(e)}")
                source_text = "Text extraction unavailable"

            # Generate target text
            try:
                with torch.no_grad():
                    target_outputs = text_model.generate(
                        input_features=inputs["input_features"],
                        tgt_lang=model_language,
                        num_beams=4,
                        max_new_tokens=1000,
                        length_penalty=0.8,
                        repetition_penalty=1.5,
                        no_repeat_ngram_size=3
                    )
                    target_text = processor.batch_decode(target_outputs, skip_special_tokens=True)[0]
                    logger.info(f"Target text: {target_text}")
            except Exception as e:
                logger.error(f"Target text generation error: {str(e)}")
                target_text = "Text extraction unavailable"

            # Generate translated audio
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    tgt_lang=model_language,
                    num_beams=5,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    length_penalty=1.0,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3
                )
            
            # Process model output
            if isinstance(outputs, tuple):
                logger.info("Processing tuple output from model")
                audio_output = outputs[0].cpu().numpy()
            else:
                logger.info("Processing tensor output from model")
                audio_output = outputs.cpu().numpy()

            if audio_output is None or audio_output.size == 0:
                raise ValueError("No audio data generated")

            if np.abs(audio_output).max() > 1.0:
                audio_output = audio_output / np.abs(audio_output).max()

        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return jsonify({'error': f'Translation failed: {str(e)}'}), 500

        # Process output audio
        try:
            output_tensor = torch.from_numpy(audio_output).unsqueeze(0)
            output_diagnostics = audio_processor.diagnostics.analyze_translation(output_tensor, model_language)
            
            if output_diagnostics:
                output_report = audio_processor.diagnostics.generate_report(output_diagnostics, model_language)
                if output_report:
                    logger.info("\n" + "="*50)
                    logger.info("Translated Audio Quality Report")
                    logger.info("="*50)
                    logger.info(f"Request ID: {request_id}")
                    logger.info(f"Target Language: {model_language}")
                    logger.info("-"*50)
                    logger.info(output_report)
                    logger.info("="*50 + "\n")
                    
                    if 'metrics' in output_diagnostics:
                        for metric, value in output_diagnostics['metrics'].items():
                            logger.debug(f"Quality Metric - {metric}: {value}")
            
            logger.info(f"Request {request_id}: Audio output processed successfully")
            
        except Exception as e:
            logger.error(f"Diagnostics error for request {request_id}: {str(e)}")
            logger.warning("Continuing with translation despite diagnostics failure")

        # Prepare response
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_output:
                temp_files.append(temp_output.name)
                
                waveform_tensor = torch.tensor(audio_output)
                if len(waveform_tensor.shape) == 1:
                    waveform_tensor = waveform_tensor.unsqueeze(0)
                
                torchaudio.save(
                    temp_output.name,
                    waveform_tensor,
                    sample_rate=16000
                )
                
                if not Path(temp_output.name).exists() or Path(temp_output.name).stat().st_size == 0:
                    raise ValueError("Failed to save translated audio")
                
                with open(temp_output.name, 'rb') as audio_file:
                    audio_data = audio_file.read()
                
                if not audio_data:
                    raise ValueError("Generated audio data is empty")
                
                response_data = {
                    'audio': base64.b64encode(audio_data).decode('utf-8'),
                    'transcripts': {
                        'source': source_text,
                        'target': target_text
                    }
                }

                response = Response(
                    json.dumps(response_data),
                    mimetype='application/json',
                    headers={
                        'Content-Type': 'application/json',
                        'Cache-Control': 'no-cache, no-store, must-revalidate',
                        'Pragma': 'no-cache',
                        'Expires': '0',
                        'Access-Control-Allow-Origin': request.headers.get('Origin', 'http://localhost:3000'),
                        'Access-Control-Allow-Methods': 'POST, OPTIONS',
                        'Access-Control-Allow-Headers': 'Content-Type',
                        'Access-Control-Allow-Credentials': 'true'
                    }
                )
                
                logger.info(f"Request {request_id}: Response prepared successfully")
                return response

        except Exception as e:
            logger.error(f"Response preparation error: {str(e)}")
            return jsonify({'error': f'Failed to prepare response: {str(e)}'}), 500

    finally:
        for temp_file in temp_files:
            cleanup_file(temp_file)
        duration = time.time() - start_time
        logger.info(f"Request {request_id} completed in {duration:.2f}s")