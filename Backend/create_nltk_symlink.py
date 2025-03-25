# create_nltk_symlink.py
import os
import nltk

# First, make sure the standard tagger is downloaded
nltk.download('averaged_perceptron_tagger')

# Find the NLTK data directory
nltk_data_dir = os.path.expanduser('~/nltk_data')

# Create the directory structure if it doesn't exist
os.makedirs(f"{nltk_data_dir}/taggers/averaged_perceptron_tagger_eng", exist_ok=True)

# Create a symbolic link from the standard tagger to the one ESPnet is looking for
source_dir = f"{nltk_data_dir}/taggers/averaged_perceptron_tagger"
target_dir = f"{nltk_data_dir}/taggers/averaged_perceptron_tagger_eng"

# Copy all files from source to target
import shutil
for item in os.listdir(source_dir):
    source_item = os.path.join(source_dir, item)
    target_item = os.path.join(target_dir, item)
    if not os.path.exists(target_item):
        if os.path.isdir(source_item):
            shutil.copytree(source_item, target_item)
        else:
            shutil.copy2(source_item, target_item)

print("Created symlink for averaged_perceptron_tagger_eng")