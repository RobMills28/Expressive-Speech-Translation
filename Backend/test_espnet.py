import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_espnet")
dddsss
def main():
    logger.info("Testing ESPnet installation")
    
    try:
        # Import basic ESPnet module
        import espnet
        logger.info(f"ESPnet version: {espnet.__version__}")
        
        # Try importing key components
        try:
            from espnet2.bin.asr_inference import Speech2Text
            logger.info("Successfully imported Speech2Text")
        except ImportError as e:
            logger.error(f"Failed to import Speech2Text: {str(e)}")
            
        try:
            from espnet2.bin.tts_inference import Text2Speech
            logger.info("Successfully imported Text2Speech")
        except ImportError as e:
            logger.error(f"Failed to import Text2Speech: {str(e)}")
            
        try:
            from espnet2.bin.mt_inference import Text2Text
            logger.info("Successfully imported Text2Text")
        except ImportError as e:
            logger.error(f"Failed to import Text2Text: {str(e)}")
            
        logger.info("ESPnet import test passed!")
    except ImportError as e:
        logger.error(f"Failed to import ESPnet: {str(e)}")

if __name__ == "__main__":
    main()