# run_translation_job.py

import os
import sys
import json
import boto3
import tempfile
from pathlib import Path
import logging
import time
import torch
import torchaudio

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

# --- Import Your Existing Logic ---
# This assumes the script is in the `Backend` folder, alongside the `services` directory.
try:
    from services.cascaded_backend import CascadedBackend
    from services.audio_processor import AudioProcessor
    logger.info("Successfully imported core translation services.")
except ImportError as e:
    logger.critical(f"Failed to import services. Ensure this script is in the 'Backend' directory. Error: {e}")
    sys.exit(1)

# --- Main Worker Logic ---
def process_job(job_message: dict):
    """
    This is the core function that processes a single translation job.
    """
    job_id = job_message.get('jobId', 'unknown-job')
    logger.info(f"[{job_id}] --- Starting job processing ---")

    s3_client = boto3.client('s3')
    dynamodb_client = boto3.resource('dynamodb')

    # Get table name from environment variable
    table_name = os.environ['DYNAMODB_TABLE_NAME']
    table = dynamodb_client.Table(table_name)
    
    try:
        # 1. Update job status to PROCESSING
        timestamp = int(time.time())
        table.update_item(
            Key={'jobId': job_id},
            UpdateExpression="set #status = :s, #updatedAt = :t",
            ExpressionAttributeNames={'#status': 'status', '#updatedAt': 'updatedAt'},
            ExpressionAttributeValues={':s': 'PROCESSING', ':t': timestamp}
        )
        logger.info(f"[{job_id}] Updated job status to PROCESSING.")
        
        # 2. Parse job details from DynamoDB (SQS message only has jobId)
        job_item = table.get_item(Key={'jobId': job_id})['Item']
        
        input_s3_key = job_item['s3Key']
        bucket_name = job_item['s3Bucket']
        target_lang = job_item['targetLanguage']
        
        # Extract userId from s3Key: "uploads/{userId}/{jobId}/filename"
        try:
            user_id = input_s3_key.split('/')[1]
        except IndexError:
            user_id = 'unknown-user'

        logger.info(f"[{job_id}] Job details: User='{user_id}', Bucket='{bucket_name}', Key='{input_s3_key}', TargetLang='{target_lang}'")

        # 3. Create a temporary, self-cleaning directory.
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            local_input_path = temp_dir_path / Path(input_s3_key).name
            
            # 4. Download the source file from S3.
            logger.info(f"[{job_id}] Downloading '{input_s3_key}' from S3...")
            s3_client.download_file(bucket_name, input_s3_key, str(local_input_path))
            
            # 5. Initialize AI models and processors.
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            backend = CascadedBackend(device=device)
            backend.initialize()
            audio_processor = AudioProcessor()
            
            # 6. Process the audio file into a tensor.
            audio_tensor = audio_processor.process_audio(str(local_input_path))

            # 7. Execute the translation.
            logger.info(f"[{job_id}] Starting translation pipeline...")
            result = backend.translate_speech(
                audio_tensor=audio_tensor,
                source_lang="eng",
                target_lang=target_lang
            )
            translated_audio_tensor = result["audio"]
            logger.info(f"[{job_id}] Translation pipeline complete.")

            # 8. Save the translated audio to a temporary file.
            final_output_path = temp_dir_path / "translated.wav"
            if translated_audio_tensor.ndim == 1:
                translated_audio_tensor = translated_audio_tensor.unsqueeze(0)
            torchaudio.save(str(final_output_path), translated_audio_tensor.cpu(), 16000)

            # 9. Upload the result back to S3.
            output_s3_key = f"results/{user_id}/{job_id}/translated_audio.wav"
            logger.info(f"[{job_id}] Uploading result to '{output_s3_key}'...")
            s3_client.upload_file(str(final_output_path), bucket_name, output_s3_key)

        # 10. Update job status to COMPLETE
        timestamp = int(time.time())
        table.update_item(
            Key={'jobId': job_id},
            UpdateExpression="set #status = :s, #updatedAt = :t, #resultS3Key = :r",
            ExpressionAttributeNames={'#status': 'status', '#updatedAt': 'updatedAt', '#resultS3Key': 'resultS3Key'},
            ExpressionAttributeValues={':s': 'COMPLETE', ':t': timestamp, ':r': output_s3_key}
        )

        logger.info(f"[{job_id}] --- Job finished successfully ---")
        return True

    except Exception as e:
        logger.error(f"[{job_id}] --- Job FAILED during execution ---", exc_info=True)
        # Update DynamoDB to reflect the failure
        timestamp = int(time.time())
        table.update_item(
            Key={'jobId': job_id},
            UpdateExpression="set #status = :s, #updatedAt = :t, #errorMessage = :e",
            ExpressionAttributeNames={'#status': 'status', '#updatedAt': 'updatedAt', '#errorMessage': 'errorMessage'},
            ExpressionAttributeValues={':s': 'FAILED', ':t': timestamp, ':e': str(e)}
        )
        return False

if __name__ == '__main__':
    # This entry point is for testing the script directly.
    # It expects the SQS message body as a JSON string from the command line.
    if len(sys.argv) < 2:
        print("Usage: python run_translation_job.py '<json_message_body>'")
        sys.exit(1)
    
    # We also need the table name for local testing
    if 'DYNAMODB_TABLE_NAME' not in os.environ:
        print("Error: DYNAMODB_TABLE_NAME environment variable must be set.")
        sys.exit(1)
        
    job_data = json.loads(sys.argv[1])
    process_job(job_data)