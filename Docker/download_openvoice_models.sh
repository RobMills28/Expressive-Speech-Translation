#!/bin/bash
# Script to download necessary model checkpoints for OpenVoice

# Exit on error
set -e

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create directories
echo -e "${BLUE}Creating directories for OpenVoice checkpoints...${NC}"
mkdir -p ../checkpoints_v2/converter
mkdir -p ../checkpoints_v2/base_speakers/EN
mkdir -p ../checkpoints_v2/base_speakers/ZH
mkdir -p ../checkpoints_v2/base_speakers/ses

# Download OpenVoice v2 checkpoints
echo -e "${YELLOW}Downloading OpenVoice v2 checkpoints...${NC}"
if [ ! -f "../checkpoints_v2/converter/config.json" ]; then
    echo "Downloading main checkpoints..."
    curl -L https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip -o openvoice_checkpoints.zip
    unzip -o openvoice_checkpoints.zip -d ../checkpoints_v2
    rm openvoice_checkpoints.zip
    echo -e "${GREEN}Successfully downloaded main OpenVoice checkpoints${NC}"
else
    echo -e "${GREEN}OpenVoice checkpoints already exist. Skipping download.${NC}"
fi

# Check if the voice embedding file exists
if [ ! -f "../checkpoints_v2/base_speakers/ses/en-us.pth" ]; then
    echo -e "${YELLOW}Speaker embedding file missing. Creating...${NC}"
    
    # Copy a default embedding from English
    if [ -f "../checkpoints_v2/base_speakers/EN/en_default_se.pth" ]; then
        echo "Using English default embedding as template..."
        cp ../checkpoints_v2/base_speakers/EN/en_default_se.pth ../checkpoints_v2/base_speakers/ses/en-us.pth
        echo -e "${GREEN}Speaker embedding created!${NC}"
    else
        echo -e "${RED}Cannot find source embedding. Download may have failed.${NC}"
        exit 1
    fi
fi

# Create compatible version of speaker embedding
echo -e "${YELLOW}Creating compatible speaker embedding with Python script...${NC}"
cd ..
python docker/create_embedding.py --source_file "checkpoints_v2/base_speakers/ses/en-us.pth" --output_file "checkpoints_v2/base_speakers/ses/en-us-new.pth"

# Backup original and replace with compatible version
if [ -f "checkpoints_v2/base_speakers/ses/en-us-new.pth" ]; then
    cp checkpoints_v2/base_speakers/ses/en-us.pth checkpoints_v2/base_speakers/ses/en-us-original.pth
    cp checkpoints_v2/base_speakers/ses/en-us-new.pth checkpoints_v2/base_speakers/ses/en-us.pth
    echo -e "${GREEN}Successfully created compatible speaker embedding!${NC}"
else
    echo -e "${RED}Failed to create compatible speaker embedding.${NC}"
    echo "Please check the Python script output for errors."
fi

echo -e "${GREEN}All OpenVoice model checkpoints downloaded and prepared successfully!${NC}"