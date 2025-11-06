# Installation Guide (Without Conda)

## Quick Install

```bash
# Install PyTorch with CUDA 11.8 first
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install all other dependencies
pip install -r requirements.txt
```

## Step-by-Step Installation

### 1. Install PyTorch (CUDA 11.8)

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118
```

**Note:** If you have a different CUDA version, adjust accordingly:
- CUDA 11.8: `--index-url https://download.pytorch.org/whl/cu118`
- CUDA 12.1: `--index-url https://download.pytorch.org/whl/cu121`
- CPU only: `pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2`

### 2. Install All Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
python -c "from CLIP import clip; print('CLIP installed')"
```

## Fixed Error

**Error:** `ValueError: Default process group has not been initialized`

**Fix:** Updated `models/model.py` to handle both distributed and non-distributed modes:

```python
# Before (caused error in non-distributed mode):
self.rank = dist.get_rank()

# After (works in both modes):
try:
    self.rank = dist.get_rank()
except RuntimeError:
    # Not in distributed mode
    self.rank = 0
```

## Troubleshooting

### Issue: PyTorch CUDA version mismatch
**Solution:** Make sure PyTorch CUDA version matches your system:
```bash
nvidia-smi  # Check CUDA version
python -c "import torch; print(torch.version.cuda)"  # Check PyTorch CUDA
```

### Issue: Package conflicts
**Solution:** Use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Issue: Out of memory during installation
**Solution:** Install packages one by one or use `--no-cache-dir`:
```bash
pip install --no-cache-dir -r requirements.txt
```

## Requirements Summary

- **Python:** 3.9+ (tested with 3.9.0)
- **PyTorch:** 2.0.1 with CUDA 11.8
- **CUDA:** 11.8+ (for GPU support)
- **Disk Space:** ~10GB (for packages + models)

## Next Steps

After installation:
1. Download pre-trained models: `./download_models.sh`
2. Prepare your data in `datasets/images/` and `datasets/images.json`
3. Run training: `cd train && python train.py [args...]`
