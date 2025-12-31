#!/usr/bin/env python3
import os
import sys
import json
import boto3
import time
import logging
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    queue_url = os.environ.get('SQS_QUEUE_URL')
    if not queue_url:
        logger.error("SQS_QUEUE_URL environment variable not set")
        sys.exit(1)
    
    sqs = boto3.client('sqs')
    logger.info(f"Worker started. Polling queue: {queue_url}")
    
    while True:
        try:
            # Poll for messages (long polling - 20 seconds)
            response = sqs.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20,
                VisibilityTimeout=3600  # 1 hour to process
            )
            
            messages = response.get('Messages', [])
            
            if not messages:
                logger.info("No messages in queue, continuing to poll...")
                continue
            
            message = messages[0]
            receipt_handle = message['ReceiptHandle']
            body = message['Body']
            
            logger.info(f"Received message: {body[:100]}...")
            
            # Run the translation job
            result = subprocess.run(
                ['python', 'run_translation_job.py', body],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("Job completed successfully")
                # Delete message from queue
                sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
            else:
                logger.error(f"Job failed: {result.stderr}")
                # Message will become visible again after VisibilityTimeout
                
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            time.sleep(5)  # Wait before retrying

if __name__ == '__main__':
    main()
