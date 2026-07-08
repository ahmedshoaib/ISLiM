# ISLiM — Keypoint-based Sign Language Translation

A compact (75M parameter) gloss-free sign language translation model that maps Indian Sign Language keypoint sequences directly to English text.

## Architecture

- **Visual Encoder** (14.5M): 94 MediaPipe Holistic landmarks (x, y, z, visibility) → two stacked Conv1D layers (k=5) → 4-layer transformer encoder (d=512)
- **Text Decoder** (T5-small, ~60M): pretrained T5-small generates English translations autoregressively via cross-attention over the 128-frame encoder output
- **Total**: ~75M parameters, fits in <5 GB VRAM

## Key Contributions

1. A **compact 75M-parameter** gloss-free keypoint-to-text architecture achieving test BLEU4 of 5.85 on iSign
2. A **multi-loss training strategy** (CE + CLIP + semantic + entropy) that prevents mode collapse on data-scarce subsets where CE-only training fails
3. **Cross-lingual transfer** to German DGS on Phoenix-2014-T with only a decoder swap, matching a 3.6B-parameter video model with 48× fewer parameters
4. A preliminary **mobile deployment study** confirming 384 ms mean inference latency on a consumer Android phone

## Quick Start

The repo ships with sample data (15 keypoint files, 5 sentences per split) so you can verify everything works before downloading the full dataset.

```bash
# Install dependencies
pip install -r requirements.txt

# Quick smoke test — verify all modules load and a single training epoch completes
python scripts/train.py --split posestitch --epochs 1 --batch_size 2 \
    --grad_accum 1 --clip_weight 1.0 --struct_weight 0.01 \
    --cool_down 0 --exp_name smoke --no_amp

# Train on full iSign with 4-loss (replace sample data with full dataset first)
python scripts/train.py --config configs/isign_4loss.yaml
```

## Environment Variable Reference

| Variable | Default | Description |
|---|---|---|
| `ISIGN_DATA_DIR` | `~/Datasets/iSign` | Root directory for the iSign dataset |
| `ISIGN_KEYPOINTS_DIR` | `$ISIGN_DATA_DIR/misc/iSign_Keypoints` | Pre-extracted .npy keypoint files |
| `ISIGN_VIDEOS_DIR` | `$ISIGN_DATA_DIR/iSign-videos_v1.1` | Raw video files |
| `POSE_STITCH_CSV_DIR` | `./data/splits` | PoseStitch CSV split files (train/val/test) |
| `SPAMOOF_CSV_DIR` | `~/Datasets/SpaMo-OF/Data` | SpaMo-OF 10K subset CSVs |
| `GERMAN_KEYPOINTS_DIR` | `~/Datasets/phoenix2014T/keypoints` | Phoenix-2014-T keypoint files |
| `GERMAN_CSV_DIR` | `~/Datasets/phoenix2014T/splits` | Phoenix-2014-T split CSVs |
| `CHECKPOINT_DIR` | `./checkpoints` | Model checkpoint save/load directory |
| `T5_MODEL_NAME` | `t5-small` | HuggingFace T5 model identifier |

## Data Preparation

### iSign Dataset

1. Request access to the iSign v1.1 dataset
2. Place the dataset at `$ISIGN_DATA_DIR` with this structure:
   ```
   $ISIGN_DATA_DIR/
     iSign_v1.1.csv
     iSign-videos_v1.1/
     misc/
       iSign_Keypoints/
         *.npy
   ```
3. Place PoseStitch CSV split files in `data/splits/`:
   ```
   data/splits/
     train.csv
     val.csv
     test.csv
   ```
   Each CSV must contain columns: `uid`, `text`, `split`, `kp_file`.

### SpaMo-OF 10K Subset

Place the 80/10/10 CSVs at `$SPAMOOF_CSV_DIR` following the SpaMo-OF data format.

### Phoenix-2014-T (German)

Place keypoint files and CSVs at `$GERMAN_KEYPOINTS_DIR` and `$GERMAN_CSV_DIR`.

## Training

```bash
# Full iSign with 4-loss
bash scripts/run.sh
# or equivalently:
python scripts/train.py --config configs/isign_4loss.yaml

# SpaMo-OF 10K subset
python scripts/train.py --config configs/spamof_4loss.yaml

# Phoenix cross-lingual transfer (CE only)
python scripts/train.py --config configs/phoenix_crosslingual.yaml
```

## Evaluation

```bash
python scripts/eval/evaluate_all.py --checkpoint checkpoints/best_model.pt
```

## Mobile Deployment

```bash
python scripts/mobile/export_torchscript.py --checkpoint checkpoints/best_model.pt
```

This produces TorchScript modules (`encoder_traced.ptl`, `t5_*.ptl`) optimized for on-device CPU inference via PyTorch Mobile.

## Citation

Will be updated upon publication.

## Directory Structure

```
anon_git/
  configs/               # YAML training configs for each experiment
    isign_4loss.yaml
    spamof_4loss.yaml
    phoenix_crosslingual.yaml
  config.py              # Central path/parameter resolution
  data/
    splits/              # PoseStitch CSV splits (not in repo)
  paper/
  scripts/
    train.py             # Main training entry point
    run.sh               # Shell wrapper for training
    eval/                # Evaluation scripts
    mobile/              # TorchScript export utilities
    viz/                 # Visualization helpers
    preprocess/          #generate keypoints
  src/
    islim/               # Core model architecture
    losses/              # Multi-loss implementations
    data/                # Dataset loaders and preprocessing
  checkpoints/           # Saved model checkpoints (not in repo)
  requirements.txt
  README.md
```
