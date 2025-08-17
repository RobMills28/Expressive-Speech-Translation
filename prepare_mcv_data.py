# Filename: examples/libritts/cosyvoice2/local/prepare_mcv_data.py
import argparse
import os
import pandas as pd
from tqdm import tqdm
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(args):
    """
    Parses a Mozilla Common Voice .tsv file to generate the Kaldi-style
    wav.scp, text, and utt2spk files required by the CosyVoice pipeline.
    """
    logging.info(f"Reading TSV file from: {args.tsv_path}")
    # Specify dtype to avoid mixed type warnings
    df = pd.read_csv(args.tsv_path, sep='\t', usecols=['path', 'sentence'], dtype={'path': str, 'sentence': str})

    # Filter out rows where the sentence is missing, as they are unusable for TTS.
    df.dropna(subset=['sentence'], inplace=True)
    logging.info(f"Found {df.shape[0]} valid utterances.")

    # If a specific number of clips is requested, subset the dataframe.
    if args.num_clips > 0 and args.num_clips < df.shape[0]:
        logging.info(f"Subsetting to the first {args.num_clips} clips for this run.")
        df = df.head(args.num_clips)

    # Ensure the destination directory exists.
    os.makedirs(args.des_dir, exist_ok=True)
    logging.info(f"Output files will be saved in: {args.des_dir}")

    # Open all output files at once.
    with open(os.path.join(args.des_dir, 'wav.scp'), 'w', encoding='utf-8') as f_wav, \
         open(os.path.join(args.des_dir, 'text'), 'w', encoding='utf-8') as f_text, \
         open(os.path.join(args.des_dir, 'utt2spk'), 'w', encoding='utf-8') as f_utt2spk:
        
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing clips"):
            clip_filename = row['path']
            clip_path = os.path.join(args.clips_dir, clip_filename)

            # Skip if the actual audio file is missing.
            if not os.path.exists(clip_path):
                continue
            
            # For fine-tuning a new language, we treat the entire dataset as
            # a single "speaker".
            speaker_id = f"{args.language}_sft_speaker"
            
            # Create a unique utterance ID from the filename.
            utterance_id = f"{args.language}_{os.path.splitext(clip_filename)[0].replace('-', '_')}"

            # Write to the respective files in the required format.
            # We need to construct the path as it will be INSIDE the container
            container_clip_path = os.path.join("/data/el/clips", clip_filename)
            f_wav.write(f"{utterance_id} {container_clip_path}\n")
            f_text.write(f"{utterance_id} {row['sentence']}\n")
            f_utt2spk.write(f"{utterance_id} {speaker_id}\n")
    
    logging.info("Successfully generated wav.scp, text, and utt2spk files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Mozilla Common Voice data for CosyVoice.")
    parser.add_argument('--tsv_path', type=str, required=True, help='Path to the validated.tsv file from MCV.')
    parser.add_argument('--clips_dir', type=str, required=True, help='Path to the clips directory containing .mp3 files.')
    parser.add_argument('--des_dir', type=str, required=True, help='Destination directory to save the output files.')
    parser.add_argument('--language', type=str, default='greek', help='A short identifier for the language (e.g., greek, french).')
    parser.add_argument('--num_clips', type=int, default=-1, help='Number of clips to process. Use -1 for all clips.')
    args = parser.parse_args()
    main(args)