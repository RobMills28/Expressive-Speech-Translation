# check_files.py
import os
import json

def check_file(path):
    """Check if a file exists and print its properties"""
    try:
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"✓ File exists: {path}")
            print(f"  Size: {size} bytes")
            
            # If it's a JSON file, try to load it
            if path.endswith('.json'):
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                    print(f"  JSON loaded successfully with {len(data)} keys")
                except Exception as e:
                    print(f"  ⚠️ Error loading JSON: {str(e)}")
        else:
            print(f"✗ File does not exist: {path}")
    except Exception as e:
        print(f"Error checking file {path}: {str(e)}")

# Check parent directory
print("=== Checking Directory Structure ===")
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"Parent directory: {parent_dir}")

# Check key directories
dirs_to_check = [
    os.path.join(parent_dir, "checkpoints_v2"),
    os.path.join(parent_dir, "checkpoints_v2", "converter"),
    os.path.join(parent_dir, "checkpoints_v2", "base_speakers"),
    os.path.join(parent_dir, "checkpoints_v2", "base_speakers", "ses")
]

for dir_path in dirs_to_check:
    if os.path.exists(dir_path):
        print(f"✓ Directory exists: {dir_path}")
        # List contents
        contents = os.listdir(dir_path)
        print(f"  Contents: {', '.join(contents)}")
    else:
        print(f"✗ Directory does not exist: {dir_path}")

# Check key files
files_to_check = [
    os.path.join(parent_dir, "checkpoints_v2", "converter", "config.json"),
    os.path.join(parent_dir, "checkpoints_v2", "converter", "checkpoint.pth"),
    os.path.join(parent_dir, "checkpoints_v2", "base_speakers", "ses", "en-us.pth")
]

print("\n=== Checking Key Files ===")
for file_path in files_to_check:
    check_file(file_path)