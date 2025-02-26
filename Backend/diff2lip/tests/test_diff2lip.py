import os
import sys
import subprocess

def test_diff2lip(video_path, audio_path, output_path):
    """Test Diff2Lip functionality with a sample video and audio."""
    try:
        # Path to the Diff2Lip checkpoint
        model_path = "diff2lip/checkpoints/archive"
        
        # Make sure paths exist
        if not os.path.exists(video_path):
            print(f"Error: Video file {video_path} does not exist")
            return False
            
        if not os.path.exists(audio_path):
            print(f"Error: Audio file {audio_path} does not exist")
            return False
            
        if not os.path.exists(model_path):
            print(f"Error: Model checkpoint {model_path} does not exist")
            return False
        
        # Command to run Diff2Lip
        command = [
            "python", "diff2lip/generate.py",
            "--model_path", model_path,
            "--face", video_path,
            "--audio", audio_path,
            "--outfile", output_path,
            "--sample_mode", "cross",  # For cross-identity lip sync
            "--time_step", "500"  # Balance between quality and speed
        ]
        
        print(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error running Diff2Lip: {result.stderr}")
            return False
            
        print(f"Successfully created lip-synced video: {output_path}")
        print(f"Output: {result.stdout}")
        return True
        
    except Exception as e:
        print(f"Error testing Diff2Lip: {str(e)}")
        return False

if __name__ == "__main__":
    # Example usage
    if len(sys.argv) < 4:
        print("Usage: python test_diff2lip.py <video_path> <audio_path> <output_path>")
        sys.exit(1)
        
    video_path = sys.argv[1]
    audio_path = sys.argv[2]
    output_path = sys.argv[3]
    
    success = test_diff2lip(video_path, audio_path, output_path)
    sys.exit(0 if success else 1)