#!/bin/bash

# Download DINO model manually if automatic download fails

DINO_URL="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
CACHE_DIR="$HOME/.cache/torch/hub/checkpoints"
CACHE_FILE="$CACHE_DIR/dino_vitbase8_pretrain.pth"

echo "Downloading DINO model..."
mkdir -p "$CACHE_DIR"

# Try wget first
if command -v wget &> /dev/null; then
    wget "$DINO_URL" -O "$CACHE_FILE"
elif command -v curl &> /dev/null; then
    curl -L "$DINO_URL" -o "$CACHE_FILE"
else
    echo "Error: Neither wget nor curl found. Please install one of them."
    exit 1
fi

if [ -f "$CACHE_FILE" ]; then
    echo "✓ DINO model downloaded successfully to: $CACHE_FILE"
    echo "File size: $(du -h "$CACHE_FILE" | cut -f1)"
else
    echo "✗ Download failed. Please check your internet connection."
    exit 1
fi
