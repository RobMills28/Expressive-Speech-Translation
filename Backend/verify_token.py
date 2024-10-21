import os
import sys
from dotenv import load_dotenv
from huggingface_hub import HfApi
from transformers import AutoConfig, AutoModel, AutoProcessor

def verify_token_and_model_access():
    # Load environment variables
    load_dotenv()
    
    token = os.getenv('HUGGINGFACE_TOKEN')
    if not token:
        print("HUGGINGFACE_TOKEN is not set in the environment.")
        return

    print(f"Python version: {sys.version}")

    # Try to import transformers
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        print("Transformers library is not installed.")
        return

    api = HfApi()
    try:
        user_info = api.whoami(token=token)
        print(f"Token is valid. Logged in as: {user_info['name']}")
    except Exception as e:
        print(f"Error verifying token: {str(e)}")
        return

    # Attempt to access SeamlessM4T model
    model_name = "facebook/seamless-m4t-medium"
    print(f"\nAttempting to access {model_name}")

    try:
        print("Loading model configuration...")
        config = AutoConfig.from_pretrained(model_name, token=token)
        print("Model configuration loaded successfully.")

        print("Loading model...")
        model = AutoModel.from_pretrained(model_name, config=config, token=token)
        print("Model loaded successfully.")

        print("Loading processor...")
        processor = AutoProcessor.from_pretrained(model_name, token=token)
        print("Processor loaded successfully.")

    except Exception as e:
        print(f"Error accessing model: {str(e)}")

    # Print .env file content (without showing the actual token)
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