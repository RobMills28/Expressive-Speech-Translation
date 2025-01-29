from flask import request, jsonify, Response
import gc
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
    """
    Handle audio translation requests with optimized parameters for CPU processing.
    
    This function processes audio files and generates translations in three steps:
    1. Source text extraction (high accuracy, no sampling)
    2. Target text generation (balanced naturalness and accuracy)
    3. Audio generation (CPU optimized with quality controls)

    Each stage uses carefully tuned parameters for the best balance of
    quality and performance on CPU hardware.
    """
    request_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
    start_time = time.time()
    temp_files = []
    source_text = "Text extraction unavailable"
    target_text = "Text extraction unavailable"
    
    try:
        logger.info(f"Starting translation request {request_id}")
        
        # Initial memory state logging
        if DEVICE.type == 'cuda':
            initial_memory = torch.cuda.memory_allocated(0) / 1024**2
            logger.info(f"Initial GPU memory allocation: {initial_memory:.2f}MB")
        
        # Resource check with early bailout
        resources_ok, resource_error = ResourceMonitor.check_resources()
        if not resources_ok:
            raise ValueError(f"Resource check failed: {resource_error}")
        
        # Request validation
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

        # Language validation
        model_language = LANGUAGE_MAP.get(target_language, target_language)
        if model_language not in LANGUAGE_CODES:
            logger.error(f"Request {request_id}: Unsupported language {target_language}")
            return jsonify({'error': f'Unsupported language: {target_language}'}), 400

        # Initialize audio processor with error handling
        try:
            audio_processor = AudioProcessor()
            if not audio_processor.diagnostics:
                logger.warning(f"Request {request_id}: AudioDiagnostics not initialized properly")
        except Exception as e:
            logger.error(f"Failed to initialize AudioProcessor: {str(e)}")
            return jsonify({'error': 'Internal processing error'}), 500

        # Audio file processing with format handling
        file_extension = Path(file.filename).suffix.lower()
        try:
            if file_extension == '.mp3':
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as wav_file:
                    temp_files.append(wav_file.name)
                    file.save(wav_file.name + '.mp3')
                    temp_files.append(wav_file.name + '.mp3')
                    sound = AudioSegment.from_mp3(wav_file.name + '.mp3')
                    # Set consistent audio parameters
                    sound = sound.set_frame_rate(SAMPLE_RATE).set_channels(1)
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

        # Audio length validation
        is_valid, error_message = audio_processor.validate_audio_length(audio_path)
        if not is_valid:
            logger.error(f"Request {request_id}: Audio validation failed - {error_message}")
            return jsonify({'error': error_message}), 400

        # Process audio with enhanced diagnostics
        try:
            logger.info(f"Request {request_id}: Beginning audio processing with diagnostics")
            
            audio, input_diagnostics = audio_processor.process_audio_enhanced(
                audio_path,
                target_language=model_language,
                return_diagnostics=True
            )
            
            # Comprehensive diagnostics logging
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

                # Check for critical quality issues
                quality_issues = []
                if input_diagnostics.get('silence_percentage', 0) > 80:
                    quality_issues.append("Audio contains too much silence")
                if input_diagnostics.get('clipping_points', 0) > 100:
                    quality_issues.append("Audio contains excessive clipping")
                if quality_issues:
                    logger.warning(f"Quality issues detected: {', '.join(quality_issues)}")
            
            audio_numpy = audio.squeeze().numpy()
            
            # Comprehensive audio validation
            if np.isnan(audio_numpy).any() or np.isinf(audio_numpy).any():
                raise ValueError("Invalid audio data detected (NaN or Inf values)")
            
            if audio_numpy.size == 0:
                raise ValueError("Empty audio data")
                
            if np.abs(audio_numpy).max() == 0:
                raise ValueError("Silent audio detected")
                
            logger.info(f"Request {request_id}: Audio processed successfully")
            
        except Exception as e:
            logger.error(f"Audio processing error: {str(e)}")
            return jsonify({'error': f'Failed to process audio: {str(e)}'}), 400
        
        # Model processing with memory optimization
        try:
            # Clear any cached memory before processing
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            inputs = processor(
                audios=audio_numpy,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                src_lang="eng",
                tgt_lang=model_language,
                padding=True,
                max_length=512000
            )
            
            logger.info(f"Input keys: {inputs.keys()}")
            
            if not inputs or not inputs.keys():
                raise ValueError("Failed to prepare model inputs")
            
            if DEVICE.type == 'cuda':
                inputs = {name: tensor.to(DEVICE) for name, tensor in inputs.items()}

            # Generate source text with high accuracy settings
            try:
                with torch.no_grad():
                    source_outputs = text_model.generate(
                        input_features=inputs["input_features"],
                        tgt_lang="eng",
                        num_beams=6,          # Higher for source text accuracy
                        do_sample=False,       # No sampling for maximum accuracy
                        max_new_tokens=8000,
                        temperature=0.2,       # Lower temperature for focused output
                        length_penalty=1.0,
                        repetition_penalty=1.5,
                        no_repeat_ngram_size=3
                    )
                    source_text = processor.batch_decode(source_outputs, skip_special_tokens=True)[0]
                    
                    # Check for common issues and retry if needed
                    if any(phrase in source_text for phrase in [
                        " H. H. H", 
                        ", the, the", 
                        " of the, of the",
                        "...",
                        "   "
                    ]):
                        logger.info("Detected potential repetition, performing second pass")
                        source_outputs = text_model.generate(
                            input_features=inputs["input_features"],
                            tgt_lang="eng",
                            num_beams=8,       # Increased for second pass
                            do_sample=False,
                            max_new_tokens=8000,
                            repetition_penalty=2.0,  # Increased
                            no_repeat_ngram_size=4,  # Increased
                            temperature=0.2,
                            length_penalty=1.5       # Adjusted
                        )
                        source_text = processor.batch_decode(source_outputs, skip_special_tokens=True)[0]
                    
                    logger.info(f"Source text: {source_text}")
                    
            except Exception as e:
                logger.error(f"Source text generation error: {str(e)}")
                source_text = "Text extraction unavailable"

            # Generate target text - Balance accuracy and naturalness
            try:
                with torch.no_grad():
                    target_outputs = text_model.generate(
                        input_features=inputs["input_features"],
                        tgt_lang=model_language,
                        num_beams=4,          # Balanced for translation
                        do_sample=True,        # Enable sampling for naturalness
                        max_new_tokens=8000,
                        temperature=0.7,       # Balanced temperature
                        length_penalty=1.0,
                        repetition_penalty=1.2,
                        no_repeat_ngram_size=3,
                        top_k=50              # Additional variety control
                    )
                    target_text = processor.batch_decode(target_outputs, skip_special_tokens=True)[0]
                    logger.info(f"Target text: {target_text}")
            except Exception as e:
                logger.error(f"Target text generation error: {str(e)}")
                target_text = "Text extraction unavailable"
            # Generate translated audio - Optimize for CPU
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        tgt_lang=model_language,
                        num_beams=2,              # Reduced for CPU efficiency
                        max_new_tokens=8000,
                        do_sample=True,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.95,
                        length_penalty=1.0,
                        repetition_penalty=1.2,
                        no_repeat_ngram_size=3
                    )

                # Handle different output types with comprehensive validation
                logger.info(f"Processing model output type: {type(outputs)}")
                if hasattr(outputs, 'waveform'):
                    logger.info("Processing waveform from SeamlessM4Tv2GenerationOutput")
                    audio_output = outputs.waveform[0].numpy()
                elif isinstance(outputs, tuple) and len(outputs) > 0:
                    logger.info("Processing tuple output from model")
                    audio_output = outputs[0].numpy()
                elif isinstance(outputs, torch.Tensor):
                    logger.info("Processing tensor output from model")
                    audio_output = outputs.numpy()
                else:
                    raise ValueError(f"Unexpected output type: {type(outputs)}")

                # Validate audio data
                if audio_output is None or audio_output.size == 0:
                    raise ValueError("No audio data generated")

                if np.isnan(audio_output).any():
                    raise ValueError("Generated audio contains NaN values")
                if np.isinf(audio_output).any():
                    raise ValueError("Generated audio contains Inf values")

                # Audio normalization with checks
                original_max = np.abs(audio_output).max()
                if original_max > 1.0:
                    logger.info(f"Normalizing audio from peak value of {original_max}")
                    audio_output = audio_output / original_max

                # Additional quality checks
                if np.abs(audio_output).max() < 0.1:
                    logger.warning("Generated audio has very low amplitude")
                if np.mean(np.abs(audio_output)) < 0.01:
                    logger.warning("Generated audio has very low average volume")

            except Exception as e:
                logger.error(f"Translation error: {str(e)}", exc_info=True)
                return jsonify({'error': f'Translation failed: {str(e)}'}), 500

            # Process output audio with quality analysis
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
                        
                        # Log detailed metrics
                        if 'metrics' in output_diagnostics:
                            for metric, value in output_diagnostics['metrics'].items():
                                logger.debug(f"Quality Metric - {metric}: {value}")
                
                            # Check for critical quality issues
                            if output_diagnostics['metrics'].get('noise_level', 0) > 0.7:
                                logger.warning("High noise level detected in output audio")
                            if output_diagnostics['metrics'].get('speech_clarity', 0) < 0.3:
                                logger.warning("Low speech clarity in output audio")
                
                logger.info(f"Request {request_id}: Audio output processed successfully")
                
            except Exception as e:
                logger.error(f"Diagnostics error for request {request_id}: {str(e)}")
                logger.warning("Continuing with translation despite diagnostics failure")

            # Prepare response with comprehensive error handling
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_output:
                    temp_files.append(temp_output.name)
                    
                    # Prepare audio tensor for saving
                    waveform_tensor = torch.tensor(audio_output)
                    if len(waveform_tensor.shape) == 1:
                        waveform_tensor = waveform_tensor.unsqueeze(0)
                    
                    # Validate tensor before saving
                    if torch.isnan(waveform_tensor).any():
                        raise ValueError("NaN values detected in output tensor")
                    if torch.isinf(waveform_tensor).any():
                        raise ValueError("Inf values detected in output tensor")
                    
                    # Save audio file
                    torchaudio.save(
                        temp_output.name,
                        waveform_tensor,
                        sample_rate=SAMPLE_RATE
                    )
                    
                    # Verify saved file
                    if not Path(temp_output.name).exists():
                        raise ValueError("Failed to save translated audio file")
                    
                    file_size = Path(temp_output.name).stat().st_size
                    if file_size == 0:
                        raise ValueError("Generated audio file is empty")
                    logger.debug(f"Saved audio file size: {file_size/1024:.2f}KB")
                    
                    # Read and validate audio data
                    with open(temp_output.name, 'rb') as audio_file:
                        audio_data = audio_file.read()
                    
                    if not audio_data:
                        raise ValueError("Failed to read generated audio data")
                    
                    # Prepare response data
                    response_data = {
                        'audio': base64.b64encode(audio_data).decode('utf-8'),
                        'transcripts': {
                            'source': source_text,
                            'target': target_text
                        }
                    }

                    # Create response with appropriate headers
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

        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return jsonify({'error': f'Translation failed: {str(e)}'}), 500

    finally:
        # Comprehensive cleanup
        try:
            # Clean up temporary files
            for temp_file in temp_files:
                cleanup_file(temp_file)
            
            # Memory cleanup
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            # Log completion and memory usage
            duration = time.time() - start_time
            logger.info(f"Request {request_id} completed in {duration:.2f}s")
            
            if DEVICE.type == 'cuda':
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
                memory_reserved = torch.cuda.memory_reserved(0) / 1024**2
                logger.info(f"Final GPU Memory - Allocated: {memory_allocated:.2f}MB, Reserved: {memory_reserved:.2f}MB")
            
            process_memory = ResourceMonitor.get_process_memory()
            logger.info(f"Process memory usage: {process_memory:.2f}MB")
            
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {str(cleanup_error)}")