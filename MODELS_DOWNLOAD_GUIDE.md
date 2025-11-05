# Pre-trained Models Download Guide

## Overview

This project uses **3 pre-trained models**. To speed up training and avoid repeated downloads, you should download them to local paths.

---

## 🔴 Required Models (Must Download)

### 1. **Stable Diffusion Inpainting** (Largest - ~4GB)

**Purpose:** Generates edited images in masked regions

**Loading Code:**
```python
# train/engine.py line 33
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_path,
    revision="fp16",
    torch_dtype=torch.float16
)
```

**Download Options:**

**Option A: HuggingFace CLI (Recommended)**
```bash
# Install huggingface-hub if needed
pip install huggingface-hub

# Download model
huggingface-cli download stabilityai/stable-diffusion-2-inpainting \
    --local-dir /home/liying/Desktop/stable-diffusion-inpainting \
    --local-dir-use-symlinks False
```

**Option B: Python Script**
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="stabilityai/stable-diffusion-2-inpainting",
    local_dir="/home/liying/Desktop/stable-diffusion-inpainting",
    local_dir_use_symlinks=False
)
```

**Option C: Manual Download**
1. Visit: https://huggingface.co/stabilityai/stable-diffusion-2-inpainting
2. Click "Files and versions" tab
3. Download all files to: `/home/liying/Desktop/stable-diffusion-inpainting/`

**Model Size:** ~4GB (fp16 version)

**Configuration:**
- Already configured in `project.py` line 25:
```python
INPAINTING_MODEL_PATH = "/home/liying/Desktop/stable-diffusion-inpainting"
```

---

### 2. **DINO ViT-Base** (~300MB)

**Purpose:** Extracts image features and attention maps for region detection

**Loading Code:**
```python
# models/model.py line 36
state_dict = torch.hub.load_state_dict_from_url(
    url="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
)
```

**Download Options:**

**Option A: Automatic (First Run)**
- Model will auto-download to `~/.cache/torch/hub/checkpoints/` on first use
- File: `dino_vitbase8_pretrain.pth`

**Option B: Manual Download**
```bash
# Create directory
mkdir -p ~/.cache/torch/hub/checkpoints/

# Download model
wget https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth \
    -P ~/.cache/torch/hub/checkpoints/
```

**Model Size:** ~300MB

**Note:** This is automatically cached by PyTorch, so you don't need to configure a path.

---

### 3. **CLIP ViT-B/16** (~600MB)

**Purpose:** Computes text-image similarity for loss calculation

**Loading Code:**
```python
# models/clip_extractor.py line 18
model = clip.load('ViT-B/16', device=device)[0]
```

**Download Options:**

**Option A: Automatic (First Run)**
- Model will auto-download to `~/.cache/clip/` on first use
- File: `ViT-B-16.pt`

**Option B: Manual Download**
```bash
# Create directory
mkdir -p ~/.cache/clip/

# Download model (requires OpenAI CLIP repository)
# Or download directly:
wget https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt \
    -P ~/.cache/clip/
```

**Model Size:** ~600MB

**Note:** This is automatically cached by the CLIP library, so you don't need to configure a path.

---

## 📊 Summary Table

| Model | Size | Auto-Download | Local Path | Config Required |
|-------|------|--------------|-----------|-----------------|
| **Stable Diffusion Inpainting** | ~4GB | ✅ Yes (HuggingFace) | `/home/liying/Desktop/stable-diffusion-inpainting` | ✅ Yes (in `project.py`) |
| **DINO ViT-Base** | ~300MB | ✅ Yes (PyTorch Hub) | `~/.cache/torch/hub/checkpoints/` | ❌ No |
| **CLIP ViT-B/16** | ~600MB | ✅ Yes (CLIP cache) | `~/.cache/clip/` | ❌ No |

**Total Size:** ~5GB

---

## 🚀 Quick Setup Script

Create a script to download all models:

```bash
#!/bin/bash
# download_models.sh

echo "Downloading pre-trained models..."

# 1. Stable Diffusion Inpainting
echo "Downloading Stable Diffusion Inpainting (~4GB)..."
mkdir -p /home/liying/Desktop/stable-diffusion-inpainting
huggingface-cli download stabilityai/stable-diffusion-2-inpainting \
    --local-dir /home/liying/Desktop/stable-diffusion-inpainting \
    --local-dir-use-symlinks False

# 2. DINO ViT-Base
echo "Downloading DINO ViT-Base (~300MB)..."
mkdir -p ~/.cache/torch/hub/checkpoints/
wget https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth \
    -P ~/.cache/torch/hub/checkpoints/

# 3. CLIP ViT-B/16
echo "Downloading CLIP ViT-B/16 (~600MB)..."
mkdir -p ~/.cache/clip/
wget https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt \
    -P ~/.cache/clip/

echo "All models downloaded successfully!"
echo "Total size: ~5GB"
```

**Usage:**
```bash
chmod +x download_models.sh
./download_models.sh
```

---

## ⚠️ Important Notes

### 1. Stable Diffusion Model
- **Must download manually** to avoid repeated downloads during training
- Already configured in `project.py`
- If you change the path, update `project.py` or use `--diffusion_model_path` argument

### 2. DINO and CLIP
- Will auto-download on first run if not present
- Cached automatically by PyTorch/CLIP
- No configuration needed
- Manual download is optional but recommended for faster first run

### 3. Network Requirements
- **Stable Diffusion**: Requires access to HuggingFace (may need VPN in some regions)
- **DINO**: Direct download from Facebook Research
- **CLIP**: Direct download from OpenAI

### 4. Disk Space
- Ensure at least **6GB** free space (5GB models + temporary files)
- Stable Diffusion is the largest component

---

## 🔧 Verification

After downloading, verify models are accessible:

```python
# Test Stable Diffusion
from diffusers import StableDiffusionInpaintPipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "/home/liying/Desktop/stable-diffusion-inpainting",
    revision="fp16",
    torch_dtype=torch.float16
)
print("✓ Stable Diffusion loaded")

# Test DINO
import torch
dino_path = torch.hub.load_state_dict_from_url(
    "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
)
print("✓ DINO loaded")

# Test CLIP
from CLIP import clip
model, _ = clip.load('ViT-B/16', device='cpu')
print("✓ CLIP loaded")
```

---

## 💡 Tips for Faster Training

1. **Pre-download Stable Diffusion** (most important - saves ~5 minutes per run)
2. **Use SSD** for model storage (faster loading)
3. **Keep models on same disk** as your dataset (avoid network transfer)
4. **Use local paths** instead of HuggingFace IDs (faster initialization)

---

## 📝 Environment Variables (Optional)

If you want to change cache directories:

```bash
# For PyTorch Hub (DINO)
export TORCH_HOME=/path/to/torch/cache

# For HuggingFace (Stable Diffusion)
export HF_HOME=/path/to/huggingface/cache

# For CLIP
# Edit CLIP/clip/clip.py to change download_root
```

---

## Summary

**Minimum Required Download:**
- ✅ **Stable Diffusion Inpainting** (~4GB) - **Must download manually**

**Optional (will auto-download but recommended):**
- ⚠️ DINO ViT-Base (~300MB) - Will auto-download
- ⚠️ CLIP ViT-B/16 (~600MB) - Will auto-download

**Total:** ~5GB for all models
