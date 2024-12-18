import os
import sys
from dotenv import load_dotenv
from huggingface_hub import HfApi
from transformers import (
    AutoConfig,
    SeamlessM4TProcessor,
    SeamlessM4Tv2Model,
    SeamlessM4TModel
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
        
        # Try loading processor
        print("\nLoading processor...")
        processor = SeamlessM4TProcessor.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True
        )
        print("Processor loaded successfully.")

        # Try loading v2 model
        print("\nAttempting to load SeamlessM4Tv2Model...")
        try:
            model_v2 = SeamlessM4Tv2Model.from_pretrained(
                model_name,
                token=token,
                trust_remote_code=True
            )
            print("Successfully loaded SeamlessM4Tv2Model")
            model = model_v2
        except Exception as e:
            print(f"Error loading SeamlessM4Tv2Model: {str(e)}")
            print("\nTrying SeamlessM4TModel instead...")
            try:
                model = SeamlessM4TModel.from_pretrained(
                    model_name,
                    token=token,
                    trust_remote_code=True
                )
                print("Successfully loaded SeamlessM4TModel")
            except Exception as e2:
                print(f"Error loading SeamlessM4TModel: {str(e2)}")
                raise Exception("Failed to load either model version")

        # Verify model can be moved to GPU if available
        if device.type == 'cuda':
            model = model.to(device)
            print("Model successfully moved to GPU")

        # Basic model verification
        print("\nVerifying model with dummy input...")
        sample_audio = torch.randn(1, 16000)  # 1 second of audio at 16kHz
        inputs = processor(
            audios=sample_audio.numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            src_lang="eng",
            tgt_lang="fra"
        )
        
        if device.type == 'cuda':
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                tgt_lang="fra",
                num_beams=1,
                max_new_tokens=50
            )
        print("Model verification successful!")

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