import os
import sys
from dotenv import load_dotenv
from huggingface_hub import HfApi
from transformers import (
    AutoConfig,
    SeamlessM4Tv2Processor,  # Changed to v2
    SeamlessM4Tv2Model,
    SeamlessM4Tv2Tokenizer,  # Added v2 tokenizer
    SeamlessM4Tv2ForSpeechToText  # Added speech-to-text model
)
import torch

def verify_token_and_model_access():
    # Load environment variables
    load_dotenv()
    
    token = os.getenv('HUGGINGFACE_TOKEN')
    if not token:
        print("HUGGINGFACE_TOKEN is not set in the environment.")
        return

    print(f"Python version: {sys.version}")
    
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    # Try to import transformers
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        print("Transformers library is not installed.")
        return

    # Verify token
    api = HfApi()
    try:
        user_info = api.whoami(token=token)
        print(f"Token is valid. Logged in as: {user_info['name']}")
    except Exception as e:
        print(f"Error verifying token: {str(e)}")
        return

    # Attempt to access SeamlessM4T model
    model_name = "facebook/seamless-m4t-v2-large"
    print(f"\nAttempting to access {model_name}")

    try:
        # Load and check configuration
        print("\nLoading model configuration...")
        config = AutoConfig.from_pretrained(model_name, token=token)
        print("Model configuration loaded successfully.")
        print(f"Model type: {config.model_type}")
        print(f"Architecture: {config.architectures[0] if config.architectures else 'Not specified'}")
        
        # Try loading v2 processor
        print("\nLoading processor...")
        processor = SeamlessM4Tv2Processor.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True
        )
        print("Processor loaded successfully.")

        # Try loading v2 tokenizer
        print("\nLoading tokenizer...")
        tokenizer = SeamlessM4Tv2Tokenizer.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True
        )
        print("Tokenizer loaded successfully.")

        # Try loading speech-to-text model
        print("\nLoading speech-to-text model...")
        text_model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
            model_name,
            token=token,
            torch_dtype=torch.float32,
            use_safetensors=True
        )
        print("Speech-to-text model loaded successfully.")

        # Try loading v2 main model
        print("\nAttempting to load SeamlessM4Tv2Model...")
        try:
            model = SeamlessM4Tv2Model.from_pretrained(
                model_name,
                token=token,
                trust_remote_code=True
            )
            print("Successfully loaded SeamlessM4Tv2Model")
        except Exception as e:
            print(f"Error loading SeamlessM4Tv2Model: {str(e)}")
            print("\nFull error traceback:")
            import traceback
            traceback.print_exc()
            raise

        # Verify models can be moved to GPU if available
        if device.type == 'cuda':
            model = model.to(device)
            text_model = text_model.to(device)
            print("Models successfully moved to GPU")

        # Basic model verification
        print("\nVerifying model with dummy input...")
        sample_audio = torch.randn(1, 16000)  # 1 second of audio at 16kHz
        inputs = processor(
            audios=sample_audio.numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            src_lang="eng",
            tgt_lang="spa"  # Testing with Spanish
        )
        
        if device.type == 'cuda':
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Test text generation
        print("\nTesting text generation...")
        with torch.no_grad():
            text_outputs = text_model.generate(
                input_features=inputs["input_features"],
                tgt_lang="spa",
                num_beams=1,
                max_new_tokens=50
            )
            decoded_text = processor.batch_decode(text_outputs, skip_special_tokens=True)[0]
            print(f"Generated text: {decoded_text}")
        
        # Test audio generation
        print("\nTesting audio generation...")
        with torch.no_grad():
            audio_outputs = model.generate(
                **inputs,
                tgt_lang="spa",
                num_beams=1,
                max_new_tokens=50
            )
        print("Model verification successful!")

        # Test language token handling
        print("\nTesting language token handling...")
        # Try to get language token IDs for verification
        try:
            spa_token = tokenizer.get_lang_token_id("spa")
            print(f"Spanish language token ID: {spa_token}")
        except Exception as e:
            print(f"Warning: Error getting language token ID: {str(e)}")
            print("This might indicate an issue with language handling")

    except Exception as e:
        print(f"\nError accessing model: {str(e)}")
        print("\nFull error traceback:")
        import traceback
        traceback.print_exc()

    # Check dependencies
    print("\nChecking required dependencies...")
    try:
        import tiktoken
        print("✓ tiktoken installed")
    except ImportError:
        print("✗ tiktoken not installed. Install with: pip install tiktoken")
    
    try:
        import sentencepiece
        print("✓ sentencepiece installed")
    except ImportError:
        print("✗ sentencepiece not installed. Install with: pip install sentencepiece")

    # Print .env file content
    print("\nContent of .env file:")
    try:
        with open('.env', 'r') as f:
            for line in f:
                if line.startswith('HUGGINGFACE_TOKEN='):
                    print('HUGGINGFACE_TOKEN=****')
                else:
                    print(line.strip())
    except FileNotFoundError:
        print(".env file not found.")

if __name__ == "__main__":
    verify_token_and_model_access()