# Batch-Processing/create_batch_manifest.py

import csv
import random
import uuid
from pathlib import Path
import argparse

def create_manifest(dataset_base_path: Path, output_manifest_file: Path):
    """
    Scans specified dataset directories, samples video files, and writes
    the list to a CSV manifest for batch processing.
    """
    DATASETS_TO_PROCESS = {
        "VoxCeleb2": {"glob_pattern": "**/*.mkv", "sample_size": 100},
        "MEAD": {"glob_pattern": "**/*.mp4", "sample_size": 100},
        "CMU-MOSEI": {"glob_pattern": "Raw/Videos/**/*.mp4", "sample_size": 100},
        "TED": {"glob_pattern": "**/*.mp4", "sample_size": 100}
    }
    
    print("--- Starting Batch Manifest Creation ---")
    all_files_to_process = []

    for name, config in DATASETS_TO_PROCESS.items():
        dataset_path = dataset_base_path / name
        print(f"\nScanning dataset: '{name}' at path '{dataset_path}'...")
        
        if not dataset_path.is_dir():
            print(f"[WARNING] Directory not found: {dataset_path}. Skipping.")
            continue
        
        found_files = list(dataset_path.glob(config["glob_pattern"]))
        print(f"Found {len(found_files)} total files.")
        
        if len(found_files) < config["sample_size"]:
            print(f"[WARNING] Found fewer files than desired. Using all {len(found_files)} files.")
            sampled_files = found_files
        else:
            sampled_files = random.sample(found_files, config["sample_size"])
        
        print(f"Sampled {len(sampled_files)} files.")
        
        for file_path in sampled_files:
            all_files_to_process.append({
                "job_id": str(uuid.uuid4()),
                "dataset": name,
                "source_video_path": str(file_path.resolve())
            })
            
    print(f"\nTotal files to process: {len(all_files_to_process)}")

    with open(output_manifest_file, 'w', newline='') as csvfile:
        fieldnames = ["job_id", "dataset", "source_video_path"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_files_to_process)
        
    print(f"\n[SUCCESS] Manifest created at: '{output_manifest_file.resolve()}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a CSV manifest for batch processing.")
    parser.add_argument("dataset_base_path", type=Path, help="The base directory where all datasets are stored.")
    parser.add_argument("--output_file", type=Path, default=Path("batch_manifest.csv"), help="The name of the output CSV file.")
    args = parser.parse_args()
    
    create_manifest(args.dataset_base_path, args.output_file)