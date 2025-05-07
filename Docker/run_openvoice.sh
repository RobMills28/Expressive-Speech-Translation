#!/bin/bash
# Script to run OpenVoice in Docker

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Show header
echo -e "${BLUE}OpenVoice Docker Setup${NC}"
echo "================================="
echo ""

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker not found. Please install Docker first.${NC}"
    echo "Visit https://docs.docker.com/get-docker/ for installation instructions."
    exit 1
else
    echo -e "${GREEN}✓ Docker installed${NC}"
fi

# Check for Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose not found. Please install Docker Compose first.${NC}"
    echo "Visit https://docs.docker.com/compose/install/ for installation instructions."
    exit 1
else
    echo -e "${GREEN}✓ Docker Compose installed${NC}"
fi

# Check for NVIDIA Docker (optional for GPU support)
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
    
    # Check for nvidia-docker
    if [ -S /var/run/docker.sock ] && docker info | grep -q "Runtimes:.*nvidia"; then
        echo -e "${GREEN}✓ NVIDIA Docker runtime is configured${NC}"
        USE_GPU=true
    else
        echo -e "${YELLOW}⚠ NVIDIA Docker runtime not detected. OpenVoice will run on CPU only.${NC}"
        echo "Visit https://github.com/NVIDIA/nvidia-docker for installation instructions."
        USE_GPU=false
    fi
else
    echo -e "${YELLOW}⚠ No NVIDIA GPU detected. OpenVoice will run on CPU only.${NC}"
    USE_GPU=false
fi

# Check for model checkpoints
echo -e "${YELLOW}Checking for OpenVoice model checkpoints...${NC}"

# Run download script if checkpoints are missing
if [ ! -d "../checkpoints_v2" ] || [ ! -f "../checkpoints_v2/converter/config.json" ]; then
    echo -e "${YELLOW}Model checkpoints not found. Downloading...${NC}"
    
    # Make script executable
    chmod +x download_openvoice_models.sh
    
    # Run the download script
    ./docker/download_openvoice_models.sh
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to download model checkpoints.${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ OpenVoice model checkpoints found${NC}"
fi

# Build and run OpenVoice container
echo -e "${YELLOW}Building OpenVoice Docker container...${NC}"
cd ..
docker-compose -f docker/docker-compose.yml build

if [ $? -ne 0 ]; then
    echo -e "${RED}Docker build failed.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker build successful${NC}"
echo -e "${YELLOW}Starting OpenVoice container...${NC}"

docker-compose -f docker/docker-compose.yml up -d

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to start Docker container.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ OpenVoice container started successfully!${NC}"
echo ""
echo -e "${BLUE}Access OpenVoice at: http://localhost:7860${NC}"
echo ""
echo "To stop the container:"
echo "docker-compose -f docker/docker-compose.yml down"
echo ""
echo "To view logs:"
echo "docker-compose -f docker/docker-compose.yml logs -f"