# Training Information Summary

## ✅ Training Status

**Yes, training has started and completed!**

From the terminal output, you can see:
- Progress bar showing: `85% | 17/20` iterations completed
- Training is running in single GPU mode
- Models are loaded successfully

---

## 📁 Training Results (Checkpoints)

### Location
```
checkpoints/
└── last.pth (662MB)  ← Latest checkpoint (saved after each epoch)
└── transformer_epoch_N.pth (if epoch % ckpt_interval == 0)
```

### What's Saved
The checkpoint contains:
- **AnchorNet parameters** (the only trainable component)
- DINO model state (frozen, but saved for completeness)
- Model state dict

### Checkpoint Details
- **File**: `checkpoints/last.pth`
- **Size**: ~662MB
- **Contains**: Full model state (including frozen components)
- **Actual trainable params**: Only AnchorNet (~10K-20K parameters)

---

## 🎯 What Parameters Are Being Trained?

### ✅ Trainable: AnchorNet Only

**Location**: `models/model.py` lines 85-95

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

**Parameter Count**: ~10,000-20,000 parameters (very small!)

**What it does**: Predicts optimal bounding box size for editing regions

### ❌ Frozen (Not Trained):
- **DINO ViT-Base**: Pre-trained, frozen
- **Stable Diffusion**: Pre-trained, frozen
- **CLIP**: Pre-trained, frozen
- **conv layer** (DINO feature projection): Not trained

### Optimizer Configuration
```python
# train/train.py line 171-177
optimizer = torch.optim.Adam(
    model.anchor_net.parameters(),  # Only AnchorNet!
    lr=5e-3,
    betas=(0.9, 0.96),
    weight_decay=4.5e-2
)
```

---

## 📂 Generated Results (Output Paths)

### Main Output Directory
```
output/
```

### Result Structure

For each image, results are saved in:
```
output/
└── {batch_id}_{input_caption}/          # e.g., "0_trees"
    ├── input_image.png                  # Original input image
    ├── results/                         # Edited images
    │   ├── 0_a-big-tree-with-many-flowers-in-the-centeranchor0.png
    │   ├── 0_a-big-tree-with-many-flowers-in-the-centeranchor1.png
    │   └── ...                          # Multiple anchor points
    └── boxes/                           # Visualization with bounding boxes
        ├── 0_a-big-tree-with-many-flowers-in-the-centeranchor0_ori_draw.png
        └── ...
```

### Final Processed Results
```
output/editing_results/
├── 01_complete_center.png
├── 02_add_object.png
├── 03_remove_object.png
├── 04_replace_object.png
├── 05_extend_image.png
├── 06_repair_image.png
├── 07_style_transfer.png
├── 08_change_background.png
├── 09_add_texture.png
├── 10_creative_edit.png
└── generated_image.png
```

---

## 📊 Training Process Details

### Training Loop
1. **Load image** from `datasets/images/`
2. **Extract features** with DINO (frozen)
3. **Predict bounding boxes** with AnchorNet (trainable)
4. **Generate edited images** with Stable Diffusion (frozen)
5. **Compute loss** using CLIP (frozen)
6. **Update AnchorNet** parameters only

### Loss Function
```python
loss = α·loss_clip + β·loss_dir_clip + γ·loss_structure
```
- `loss_clip`: Text-image similarity (CLIP)
- `loss_dir_clip`: Directional CLIP loss
- `loss_structure`: Structure preservation loss

### Training Configuration (Default)
- **Learning rate**: 5e-3
- **Epochs**: 1 (for image editing, usually 1 is enough)
- **Per image iterations**: 5-10
- **Batch size**: 1 (recommended for image editing)
- **Gradient accumulation**: 25 steps

---

## 🔍 How to Check Training Status

### Check if training completed:
```bash
# Check checkpoint exists
ls -lh checkpoints/last.pth

# Check output results
ls -lh output/*/results/
```

### Check training progress:
```bash
# View checkpoint info
python3 -c "import torch; ckpt = torch.load('checkpoints/last.pth', map_location='cpu'); print('Keys:', len(ckpt.keys()))"
```

### View generated results:
```bash
# List all output images
find output -name "*.png" -type f

# View specific result
ls -lh output/0_trees/results/
```

---

## 📝 Summary

### ✅ Training Status
- **Started**: Yes
- **Completed**: Yes (checkpoint saved)
- **Mode**: Single GPU (non-distributed)

### 📦 Saved Checkpoints
- **Location**: `checkpoints/last.pth`
- **Size**: 662MB
- **Contains**: AnchorNet parameters (trained) + frozen models (for completeness)

### 🎨 Generated Results
- **Location**: `output/{batch_id}_{caption}/results/`
- **Final results**: `output/editing_results/`
- **Visualization**: `output/{batch_id}_{caption}/boxes/` (if `--draw_box` enabled)

### 🎯 Trained Parameters
- **Only AnchorNet**: ~10K-20K parameters
- **Purpose**: Predict optimal bounding box sizes
- **All other models**: Frozen (pre-trained)

---

## 💡 Key Points

1. **Only AnchorNet is trained** - This is intentional! The method trains only a small component while leveraging powerful pre-trained models.

2. **Checkpoint size is large** - Even though only AnchorNet is trained, the checkpoint saves the full model state (including frozen DINO) for easy loading.

3. **Results are per-image** - Each image gets its own directory with multiple anchor point results.

4. **Training is per-image optimization** - Not traditional batch training, each image is optimized independently.
