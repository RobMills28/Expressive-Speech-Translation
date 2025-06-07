# Docker/test_imports.py
import sys
import os
import logging
from pathlib import Path
import time # For a small delay if needed

# Basic logging for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("cosyvoice_direct_test")

logger.info("--- Starting Direct CosyVoice Test Script ---")
logger.info(f"Python version: {sys.version}")
logger.info(f"Current sys.path: {sys.path}")
logger.info(f"Current PYTHONPATH env var: {os.getenv('PYTHONPATH')}")
logger.info(f"Current working directory: {os.getcwd()}")

# Explicitly add CosyVoice and its submodules to sys.path if needed
# This mirrors what ENV PYTHONPATH tries to do and what cosyvoice_api.py does
cosyvoice_base_path = Path("/app/CosyVoice").resolve()
matcha_tts_path = (cosyvoice_base_path / "third_party/Matcha-TTS").resolve()

if str(cosyvoice_base_path) not in sys.path:
    sys.path.insert(0, str(cosyvoice_base_path)) # Insert at beginning
    logger.info(f"Manually added to sys.path: {cosyvoice_base_path}")

if str(matcha_tts_path) not in sys.path:
    sys.path.insert(0, str(matcha_tts_path)) # Insert at beginning
    logger.info(f"Manually added to sys.path: {matcha_tts_path}")

logger.info(f"Updated sys.path: {sys.path}")

# Attempt to import and initialize
MODEL_PATH_TEST = "/app/CosyVoice/pretrained_models/CosyVoice2-0.5B"
test_model_instance = None

logger.info("Attempting to import CosyVoice2 from cosyvoice.cli.cosyvoice...")
try:
    from cosyvoice.cli.cosyvoice import CosyVoice2
    logger.info("SUCCESS: 'from cosyvoice.cli.cosyvoice import CosyVoice2' successful.")

    logger.info(f"Attempting to instantiate CosyVoice2 model from: {MODEL_PATH_TEST}")
    logger.info("This may take several minutes...")
    
    start_time = time.time()
    test_model_instance = CosyVoice2(MODEL_PATH_TEST, load_jit=False, load_trt=False, load_vllm=False, fp16=False)
    load_time = time.time() - start_time

    if test_model_instance is not None and hasattr(test_model_instance, 'sample_rate'):
        logger.info(f"SUCCESS: CosyVoice2 model instantiated successfully in {load_time:.2f} seconds. Model sample rate: {test_model_instance.sample_rate}")
    else:
        logger.error(f"FAILURE: CosyVoice2 instantiation returned None or invalid object after {load_time:.2f} seconds.")

except ImportError as mnfe:
    logger.error(f"IMPORT ERROR: {mnfe}", exc_info=True)
except FileNotFoundError as fnfe:
    logger.error(f"FILE NOT FOUND ERROR during model init: {fnfe}", exc_info=True)
    logger.error("Check that model paths are correct and models were downloaded during Docker build.")
except Exception as e:
    logger.error(f"OTHER ERROR during import or model instantiation: {e}", exc_info=True)

logger.info("--- Direct CosyVoice Test Script Finished ---")