#!/bin/bash

# Pre-trained Models Download Script
# This script downloads all required pre-trained models for faster training

set -e  # Exit on error

echo "=========================================="
echo "Pre-trained Models Download Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
STABLE_DIFFUSION_PATH="$HOME/.cache/stable-diffusion-inpainting"
TORCH_CACHE="$HOME/.cache/torch/hub/checkpoints"
CLIP_CACHE="$HOME/.cache/clip"

# # 1. Stable Diffusion Inpainting (~4GB)
# echo -e "${YELLOW}[1/3] Downloading Stable Diffusion Inpainting (~4GB)...${NC}"
# echo "This may take 10-30 minutes depending on your internet speed..."
# mkdir -p "$STABLE_DIFFUSION_PATH"

# # Check if huggingface-cli is available
# if command -v huggingface-cli &> /dev/null; then
#     echo "Using HuggingFace CLI..."
#     huggingface-cli download stabilityai/stable-diffusion-2-inpainting \
#         --local-dir "$STABLE_DIFFUSION_PATH" \
#         --local-dir-use-symlinks False
# elif command -v python3 &> /dev/null; then
#     echo "Using Python script..."
#     python3 << EOF
# from huggingface_hub import snapshot_download
# import os
# snapshot_download(
#     repo_id="stabilityai/stable-diffusion-2-inpainting",
#     local_dir="$STABLE_DIFFUSION_PATH",
#     local_dir_use_symlinks=False
# )
# print("Download complete!")
# EOF
# else
#     echo -e "${RED}Error: Neither huggingface-cli nor python3 found.${NC}"
#     echo "Please install huggingface-hub: pip install huggingface-hub"
#     echo "Or manually download from: https://huggingface.co/stabilityai/stable-diffusion-2-inpainting"
#     exit 1
# fi

# if [ -d "$STABLE_DIFFUSION_PATH" ] && [ "$(ls -A $STABLE_DIFFUSION_PATH)" ]; then
#     echo -e "${GREEN}✓ Stable Diffusion Inpainting downloaded successfully!${NC}"
# else
#     echo -e "${RED}✗ Stable Diffusion download failed!${NC}"
#     exit 1
# fi

# echo ""

# 2. DINO ViT-Base (~300MB)
echo -e "${YELLOW}[2/3] Downloading DINO ViT-Base (~300MB)...${NC}"
mkdir -p "$TORCH_CACHE"

if [ ! -f "$TORCH_CACHE/dino_vitbase8_pretrain.pth" ]; then
    wget -q --show-progress https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth \
        -P "$TORCH_CACHE"
    echo -e "${GREEN}✓ DINO ViT-Base downloaded successfully!${NC}"
else
    echo -e "${GREEN}✓ DINO ViT-Base already exists (skipping)${NC}"
fi

echo ""

# 3. CLIP ViT-B/16 (~600MB)
echo -e "${YELLOW}[3/3] Downloading CLIP ViT-B/16 (~600MB)...${NC}"
mkdir -p "$CLIP_CACHE"

if [ ! -f "$CLIP_CACHE/ViT-B-16.pt" ]; then
    wget -q --show-progress https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt \
        -P "$CLIP_CACHE"
    echo -e "${GREEN}✓ CLIP ViT-B/16 downloaded successfully!${NC}"
else
    echo -e "${GREEN}✓ CLIP ViT-B/16 already exists (skipping)${NC}"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}All models downloaded successfully!${NC}"
echo "=========================================="
echo ""
echo "Model locations:"
echo "  - Stable Diffusion: $STABLE_DIFFUSION_PATH"
echo "  - DINO: $TORCH_CACHE/dino_vitbase8_pretrain.pth"
echo "  - CLIP: $CLIP_CACHE/ViT-B-16.pt"
echo ""
echo "Total size: ~5GB"
echo ""
echo "You can now run training without waiting for downloads!"
