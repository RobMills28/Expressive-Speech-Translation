# Docker/test_imports.py (Modified for CosyVoice-300M test)
import sys
import os
import logging
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("cosyvoice_direct_test")

logger.info("--- Starting Direct CosyVoice Test Script (Testing CosyVoice-300M) ---")
logger.info(f"Python version: {sys.version}")
logger.info(f"Current sys.path (initial): {sys.path}")
logger.info(f"Current PYTHONPATH env var: {os.getenv('PYTHONPATH')}")
logger.info(f"Current working directory: {os.getcwd()}")

cosyvoice_base_path = Path("/app/CosyVoice").resolve()
matcha_tts_path = (cosyvoice_base_path / "third_party/Matcha-TTS").resolve()

if str(matcha_tts_path) not in sys.path:
    sys.path.insert(0, str(matcha_tts_path))
    logger.info(f"Manually added to sys.path (first): {matcha_tts_path}")
if str(cosyvoice_base_path) not in sys.path:
    sys.path.insert(0, str(cosyvoice_base_path))
    logger.info(f"Manually added to sys.path (first): {cosyvoice_base_path}")

logger.info(f"Current sys.path (updated for CosyVoice): {sys.path}")

try:
    logger.info("Attempting to import 'cosyvoice.cli.cosyvoice' module...")
    import cosyvoice.cli.cosyvoice as cli_cosyvoice_module
    logger.info("SUCCESS: 'import cosyvoice.cli.cosyvoice' as module successful.")
    
    logger.info("Attempting to access CosyVoice (parent) class from the imported module...")
    CosyVoice_parent_class = cli_cosyvoice_module.CosyVoice # <<<< Use parent class
    logger.info(f"SUCCESS: Accessed CosyVoice (parent) class: {CosyVoice_parent_class}")

    MODEL_PATH_300M = "/app/CosyVoice/pretrained_models/CosyVoice-300M" # <<<< Path to 300M model
    logger.info(f"Attempting to instantiate CosyVoice (parent) model from: {MODEL_PATH_300M}")
    logger.info("This may take several minutes if it reaches here...")
    
    start_time = time.time()
    # Instantiate the PARENT CosyVoice class with the 300M model path
    test_model_instance = CosyVoice_parent_class(MODEL_PATH_300M, fp16=False, load_jit=False, load_trt=False)
    load_time = time.time() - start_time

    if test_model_instance is not None and hasattr(test_model_instance, 'sample_rate'):
        logger.info(f"SUCCESS: CosyVoice (300M) model instantiated successfully in {load_time:.2f} seconds. Model sample rate: {test_model_instance.sample_rate}")
    else:
        logger.error(f"FAILURE: CosyVoice (300M) instantiation returned None or invalid object after {load_time:.2f} seconds.")

except ImportError as mnfe:
    logger.error(f"IMPORT ERROR: {mnfe}", exc_info=True)
except FileNotFoundError as fnfe:
    logger.error(f"FILE NOT FOUND ERROR during model init: {fnfe}", exc_info=True)
    logger.error("Check that model paths are correct and models were downloaded during Docker build.")
except Exception as e:
    logger.error(f"OTHER ERROR during import or model instantiation: {e}", exc_info=True)

logger.info("--- Direct CosyVoice Test Script (Testing CosyVoice-300M) Finished ---")