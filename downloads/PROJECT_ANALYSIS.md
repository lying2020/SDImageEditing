# Project Comprehensive Analysis

## 📋 What is This Project?

This is the **official implementation of "Text-Driven Image Editing via Learnable Regions" (CVPR 2024)**.

**Core Functionality:**
- **Text-driven image editing** - Edit images using only text prompts
- **Automatic region detection** - No need for manual masks or sketches
- **Bounding box generation** - Automatically finds and localizes regions to edit
- **High-fidelity editing** - Preserves image structure while applying edits

**Key Innovation:**
Instead of requiring users to manually specify which regions to edit, this method learns to automatically identify the optimal editing regions based on the text prompt.

---

## 🏗️ Architecture Overview

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│                    Input: Image + Text Prompt            │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  1. DINO Vision Transformer (Pre-trained, Frozen)       │
│     - Extract image features                            │
│     - Generate attention maps                            │
│     - Identify potential editing regions                │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  2. AnchorNet (Trainable Neural Network)                │
│     - Takes DINO features as input                     │
│     - Predicts optimal bounding box size                 │
│     - This is the ONLY trainable component!              │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  3. Stable Diffusion Inpainting (Pre-trained, Frozen)   │
│     - Receives image + mask + text prompt              │
│     - Generates edited image in masked region           │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  4. CLIP (Pre-trained, Frozen)                          │
│     - Computes text-image similarity                    │
│     - Used for loss calculation during training          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Output: Edited Image                        │
└─────────────────────────────────────────────────────────┘
```

### Model Details

**RGN (Region Generation Network) Class:**
- **DINO ViT-Base**: Pre-trained, extracts image features (768-dim)
- **AnchorNet**: Small CNN that predicts bounding box size
  - Input: ROI-aligned features from DINO
  - Output: Probability distribution over bounding box sizes
  - Architecture: Conv2d → ReLU → Conv2d → ReLU → Flatten → Linear → ReLU → Linear → LogSoftmax
  - **Only ~10K parameters!**

**What Gets Trained:**
- ✅ **Only AnchorNet parameters** (see `configure_optimizers` in train.py line 165)
- ❌ DINO: Frozen (pre-trained)
- ❌ Stable Diffusion: Frozen (pre-trained)
- ❌ CLIP: Frozen (pre-trained)

---

## 🔄 Workflow

### Training Phase

1. **Load Image & Prompts**
   - Input image from `datasets/images/`
   - Original caption (e.g., "trees")
   - Editing prompt (e.g., "a big tree with many flowers in the center")

2. **Extract Features with DINO**
   - DINO processes image → 32×32 feature map
   - Generates attention maps to identify important regions

3. **Sample Anchor Points**
   - Based on attention maps, sample N points (default: 9)
   - These are potential centers of editing regions

4. **Predict Bounding Box Size with AnchorNet**
   - For each anchor point, extract ROI features
   - AnchorNet predicts optimal bounding box size
   - Generate bounding boxes around each anchor point

5. **Generate Edited Images**
   - Create masks from bounding boxes
   - Stable Diffusion inpainting generates edited images in masked regions

6. **Compute Loss**
   - **CLIP Loss**: Text-image similarity (editing prompt vs edited image)
   - **Directional CLIP Loss**: Direction of change (source→target embedding direction)
   - **Structure Loss**: Preserve original image structure
   - **Total Loss** = α·CLIP + β·Directional + γ·Structure

7. **Backpropagate & Update**
   - Only AnchorNet parameters are updated
   - Gradient accumulation for effective larger batch size

### Inference Phase

1. Same steps 1-4 as training
2. Generate edited images (no gradient computation)
3. Save results to `output/` directory

---

## 🚀 How to Run

### Prerequisites

```bash
# 1. Environment Setup
conda create -n LearnableRegion python==3.9 -y
conda activate LearnableRegion

# 2. Install PyTorch (CUDA 11.8)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# 3. Install Dependencies
conda env update --file enviroment.yaml
```

### Prepare Data

**Option 1: Single Image**
- Place image in `datasets/images/`
- Use command-line arguments for caption and editing prompt

**Option 2: Multiple Images (Recommended)**
- Place images in `datasets/images/`
- Create/edit `datasets/images.json`:
```json
{
  "1.png": ["trees", "a big tree with many flowers in the center"],
  "2.png": ["coffee", "soft drink"]
}
```

### Run Training/Editing

**From project root directory:**

```bash
cd train
torchrun --nnodes=1 --nproc_per_node=1 train.py \
    --image_dir_path ../datasets/images/ \
    --output_dir ../output/ \
    --json_file ../datasets/images.json \
    --diffusion_model_path 'stabilityai/stable-diffusion-2-inpainting' \
    --draw_box \
    --lr 5e-3 \
    --max_window_size 15 \
    --per_image_iteration 10 \
    --epochs 1 \
    --num_workers 8 \
    --seed 42 \
    --pin_mem \
    --point_number 9 \
    --batch_size 1 \
    --save_path ../checkpoints/
```

**Key Points:**
- First run will download Stable Diffusion model (~4GB)
- Training happens per image (not typical batch training)
- Each image takes ~4 minutes on RTX 8000 GPU
- Results saved in `output/`

---

## 🎯 What Parameters Are Trained?

### Trainable Parameters: AnchorNet Only

**Location:** `models/model.py` lines 43-53

```python
self.anchor_net = nn.Sequential(
    nn.Conv2d(len(self.box)*emb_dim, emb_dim2, 1),  # Conv layer 1
    nn.ReLU(),
    nn.Conv2d(emb_dim2, 4, 1),                       # Conv layer 2
    nn.ReLU(),
    nn.Flatten(1),
    nn.Linear(16*16, 4),                              # Linear layer 1
    nn.ReLU(),
    nn.Linear(4, len(self.box)),                      # Linear layer 2
    nn.LogSoftmax(dim=1)
)
```

**Parameter Count:**
- `emb_dim = 12`, `emb_dim2 = 8`
- `len(self.box) = (max_window_size - 4) // 2 + 1` (e.g., if max_window_size=10, then 4,6,8,10 → 4 boxes)
- Total parameters: ~10,000-20,000 (very small!)

**What AnchorNet Does:**
- Input: ROI-aligned features from DINO (shape: `[batch×num_points, emb_dim×num_box_sizes, 8, 8]`)
- Output: Probability distribution over bounding box sizes
- Uses Gumbel-Softmax for differentiable sampling

### Training Configuration

**Optimizer:** Adam
- Learning rate: `--lr` (default: 5e-3)
- Betas: (0.9, 0.96)
- Weight decay: 4.5e-2

**Loss Function:**
```python
loss = α·loss_clip + β·loss_dir_clip + γ·loss_structure
```
- `loss_clip`: CLIP text-image similarity
- `loss_dir_clip`: Directional CLIP (editing direction)
- `loss_structure`: Structure preservation (self-similarity)

**Training Loop:**
- Per-image optimization (not batch-based)
- Each image trained for `per_image_iteration` steps (default: 5-10)
- Gradient accumulation: `accum_grad` steps (default: 25)
- Effective batch size = `batch_size × accum_grad`

---

## 📊 Key Hyperparameters

### Critical for Training

| Parameter | Default | Description | Impact |
|-----------|---------|-------------|--------|
| `--lr` | 5e-3 | Learning rate | Higher = faster but unstable |
| `--per_image_iteration` | 5 | Training steps per image | More = better quality, slower |
| `--max_window_size` | 10 | Max bounding box size (pixels) | Larger = bigger edit regions |
| `--point_number` | 9 | Number of anchor points | More = better localization |
| `--loss_alpha` | 1 | CLIP loss weight | Higher = more text alignment |
| `--loss_beta` | 1 | Directional loss weight | Higher = stronger editing direction |
| `--loss_gamma` | 1 | Structure loss weight | Higher = more structure preservation |

### Performance Tuning

| Parameter | Default | Description | Impact |
|-----------|---------|-------------|--------|
| `--batch_size` | 8 | Batch size | Usually 1 for image editing |
| `--accum_grad` | 25 | Gradient accumulation | Effective batch = batch_size × accum_grad |
| `--num_workers` | 16 | Data loading processes | More = faster loading, more memory |
| `--epochs` | 10 | Training epochs | Usually 1-2 for image editing |

---

## 💡 Key Insights

### Why This Approach Works

1. **Small Trainable Component**: Only AnchorNet is trained (~10K params), making training fast and stable
2. **Pre-trained Backbones**: Leverages powerful pre-trained models (DINO, SD, CLIP)
3. **Per-Image Optimization**: Each image gets its own optimization, adapting to specific content
4. **Multi-Modal Loss**: CLIP + Directional + Structure losses ensure good edits

### Training Characteristics

- **Not Traditional Batch Training**: Each image is optimized independently
- **Fast Convergence**: Usually 1 epoch is enough (5-10 iterations per image)
- **Low Memory**: Only small AnchorNet needs gradients
- **GPU Requirements**: Need GPU for Stable Diffusion inference

### What You're Actually Training

You're **NOT training a diffusion model** or **NOT training a vision transformer**. You're training a **small CNN (AnchorNet)** to predict the optimal bounding box size for editing regions, given DINO features.

Think of it as:
- **DINO** = "What regions are important?" (attention maps)
- **AnchorNet** = "How big should the edit region be?" (learned)
- **Stable Diffusion** = "Generate the edit" (pre-trained)
- **CLIP** = "Is the edit good?" (loss calculation)

---

## 📁 Directory Structure

```
SDImageEditing/
├── train/                    # Training scripts
│   ├── train.py              # Main training script
│   ├── engine.py             # Diffusion model wrapper
│   ├── vis.py                # Visualization functions
│   └── train_json.sh         # Batch processing script
├── models/                   # Model definitions
│   ├── model.py              # RGN class (main model)
│   ├── clip_extractor.py     # CLIP wrapper
│   └── vision_transformer.py # DINO ViT
├── utils/                    # Utilities
│   ├── util.py               # Dataset classes
│   ├── util2.py              # Text processing
│   └── post_process.py      # Post-processing
├── datasets/                 # Data
│   ├── images/               # Input images
│   └── images.json          # Image-prompt mapping
├── checkpoints/              # Saved models (AnchorNet)
├── output/                   # Edited images
└── project.py               # Path configuration
```

---

## 🔧 Common Issues & Solutions

### Issue: Out of Memory
- **Solution**: Reduce `--batch_size` to 1, reduce `--max_window_size`

### Issue: Slow Training
- **Solution**: Reduce `--per_image_iteration`, reduce `--point_number`

### Issue: Poor Edit Quality
- **Solution**: Increase `--per_image_iteration`, increase `--max_window_size`, adjust loss weights

### Issue: Model Download Fails
- **Solution**: Use `--diffusion_model_path` with local path, or set HuggingFace token

---

## 📝 Summary

**What this project does:**
- Text-driven image editing with automatic region detection

**What gets trained:**
- Only AnchorNet (~10K parameters) - a small CNN that predicts bounding box sizes

**How to run:**
1. Setup environment
2. Prepare images and JSON config
3. Run `train.py` with appropriate arguments
4. Results in `output/` directory

**Key point:**
This is **not** end-to-end training. It's a clever approach that trains only a small component (AnchorNet) while leveraging powerful pre-trained models for everything else.
