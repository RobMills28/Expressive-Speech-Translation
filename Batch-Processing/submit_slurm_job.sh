#!/bin/bash
#SBATCH --job-name=mcf_batch_translation
#SBATCH --output=slurm_logs/mcf_batch_%A_%a.out  # %A=main job ID, %a=array task ID
#SBATCH --error=slurm_logs/mcf_batch_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH -p cs
#SBATCH -q cspg
#SBATCH --array=1-400%10 # Process 400 jobs, but only run 10 at a time

# --- Setup ---
echo "======================================================"
echo "Job Started: $(date)"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Slurm Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $(hostname)"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "======================================================"

# Create the log directory if it doesn't exist
mkdir -p slurm_logs

# Activate your Conda environment
# This line is crucial for Slurm jobs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch

# --- Job Logic ---

# Define the paths relative to where you submit the job from (your project root)
PROJECT_DIR=$(pwd)
MANIFEST_FILE="$PROJECT_DIR/Batch-Processing/batch_manifest.csv"
OUTPUT_DIR="$PROJECT_DIR/hpc_outputs/mcf_results"

# Create the output directory
mkdir -p "$OUTPUT_DIR"

# Get the specific video path for this array task from the manifest file
# This command reads the Nth line of the CSV (plus one for the header)
VIDEO_PATH=$(awk -F, "NR==$(($SLURM_ARRAY_TASK_ID + 1)) {print \$3}" "$MANIFEST_FILE")

# Check if the video path was successfully extracted
if [ -z "$VIDEO_PATH" ]; then
    echo "Error: Could not find a video path for task ID $SLURM_ARRAY_TASK_ID in $MANIFEST_FILE"
    exit 1
fi

echo "Processing video: $VIDEO_PATH"

# Run the Python script for this single job
# Pass the video path, output directory, and target language
python "$PROJECT_DIR/Batch-Processing/run_batch_job.py" "$VIDEO_PATH" "$OUTPUT_DIR" --lang "fra"

echo "======================================================"
echo "Job Finished: $(date)"
echo "======================================================"