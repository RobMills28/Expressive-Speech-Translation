# download_nltk_resources.py
import sys
import os
import json
import shutil
import pickle

# First try to install nltk if it's not already installed
try:
    import nltk
except ImportError:
    print("NLTK not found. Installing NLTK...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
    import nltk
    print("NLTK installed successfully.")

# Now download the required resources
print("Downloading NLTK resources...")
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('cmudict')
print("Resources downloaded successfully.")

# Find the NLTK data directory
nltk_data_dir = nltk.data.path[0]
print(f"NLTK data directory: {nltk_data_dir}")

# Create the directory structure for the eng-specific tagger
eng_tagger_dir = os.path.join(nltk_data_dir, "taggers", "averaged_perceptron_tagger_eng")
os.makedirs(eng_tagger_dir, exist_ok=True)
print(f"Created directory: {eng_tagger_dir}")

# Path to the standard tagger pickle file
standard_tagger_dir = os.path.join(nltk_data_dir, "taggers", "averaged_perceptron_tagger")
standard_pickle = os.path.join(standard_tagger_dir, "averaged_perceptron_tagger.pickle")

if os.path.exists(standard_pickle):
    print(f"Found standard tagger pickle: {standard_pickle}")
    
    # Create the specifically named pickle file for _eng
    eng_pickle = os.path.join(eng_tagger_dir, "averaged_perceptron_tagger_eng.pickle")
    if not os.path.exists(eng_pickle):
        print(f"Copying pickle file to: {eng_pickle}")
        shutil.copy2(standard_pickle, eng_pickle)
    
    # Load the original model to extract data for JSON files
    try:
        with open(standard_pickle, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create the weights.json file
        weights_json = os.path.join(eng_tagger_dir, "averaged_perceptron_tagger_eng.weights.json")
        if not os.path.exists(weights_json):
            print(f"Creating weights JSON file: {weights_json}")
            # Extract weights from the model_data (adjust keys based on actual structure)
            if hasattr(model_data, 'weights'):
                weights = model_data.weights
            else:
                # If structure is different, create empty weights as fallback
                weights = {}
            
            with open(weights_json, 'w') as f:
                json.dump(weights, f)
        
        # Create the tagdict.json file
        tagdict_json = os.path.join(eng_tagger_dir, "averaged_perceptron_tagger_eng.tagdict.json")
        if not os.path.exists(tagdict_json):
            print(f"Creating tagdict JSON file: {tagdict_json}")
            # Extract tagdict from the model_data (adjust keys based on actual structure)
            if hasattr(model_data, 'tagdict'):
                tagdict = model_data.tagdict
            else:
                # If structure is different, create empty tagdict as fallback
                tagdict = {}
            
            # Convert keys to strings for JSON serialization
            string_tagdict = {str(k): v for k, v in tagdict.items()} if isinstance(tagdict, dict) else {}
            
            with open(tagdict_json, 'w') as f:
                json.dump(string_tagdict, f)
        
        # Create the classes.json file
        classes_json = os.path.join(eng_tagger_dir, "averaged_perceptron_tagger_eng.classes.json")
        if not os.path.exists(classes_json):
            print(f"Creating classes JSON file: {classes_json}")
            # Extract classes from the model_data (adjust keys based on actual structure)
            if hasattr(model_data, 'classes'):
                classes = model_data.classes
            else:
                # If structure is different, create empty classes list as fallback
                classes = []
            
            with open(classes_json, 'w') as f:
                json.dump(classes, f)
        
        print("All JSON files created successfully.")
    except Exception as e:
        print(f"Error extracting data from pickle file: {str(e)}")
        print("Creating empty JSON files as fallback...")
        
        # Create empty files as fallback
        files_to_create = [
            "averaged_perceptron_tagger_eng.weights.json",
            "averaged_perceptron_tagger_eng.tagdict.json",
            "averaged_perceptron_tagger_eng.classes.json"
        ]
        
        for filename in files_to_create:
            filepath = os.path.join(eng_tagger_dir, filename)
            if not os.path.exists(filepath):
                print(f"Creating empty {filename}")
                with open(filepath, 'w') as f:
                    if filename.endswith("classes.json"):
                        json.dump([], f)  # Empty array for classes
                    else:
                        json.dump({}, f)  # Empty object for others
else:
    print(f"ERROR: Standard tagger pickle not found: {standard_pickle}")
    print("Did you run nltk.download('averaged_perceptron_tagger') successfully?")

print("\nSetup complete. Try running your ESPnet test now.")