import os
from huggingface_hub import HfApi

def verify_token():
    token = os.getenv('HUGGINGFACE_TOKEN')
    if not token:
        print("HUGGINGFACE_TOKEN is not set in the environment.")
        return

    api = HfApi()
    try:
        user_info = api.whoami(token=token)
        print(f"Token is valid. Logged in as: {user_info['name']}")
    except Exception as e:
        print(f"Error verifying token: {str(e)}")

if __name__ == "__main__":
    verify_token()