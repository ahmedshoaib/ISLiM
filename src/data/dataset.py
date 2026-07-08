"""
Unified dataset for sign language translation.

Supports two sources:
  PoseStitchDataset — fixed 90/5/5 split (seed=42) from pre-processed CSVs
  SpaMoOFDataset   — SpaMo-OF 10K subset with PoseStitch preprocessing applied
  IndexedDataset   — wrapper that adds row_idx for correct SemanticLoss indexing
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import posestitch_preprocess
import config as cfg


_KP_CANDIDATES = [
    cfg.ISIGN_KEYPOINTS_DIR,
    Path("~/Datasets/iSign/misc/iSign_Keypoints").expanduser(),
    Path("/home/shoaib/phd/Datasets/iSign/misc/iSign_Keypoints"),
]


def _find_kp_dir():
    for path in _KP_CANDIDATES:
        if isinstance(path, Path) and path.is_dir():
            return path
        if isinstance(path, str) and Path(path).is_dir():
            return Path(path)
    return _KP_CANDIDATES[0]


KP_DIR = _find_kp_dir()


def _normalize_keypoints(keypoints, max_frames):
    x, y, z = keypoints[:, :, 0:1], keypoints[:, :, 1:2], keypoints[:, :, 2:3]
    for arr in [x, y, z]:
        rng = arr.max() - arr.min()
        if rng > 1e-8:
            arr[...] = (arr - arr.min()) / rng
        else:
            arr[...] = arr - arr.min()
    keypoints[:, :, :3] = np.concatenate([x, y, z], axis=2)

    T = keypoints.shape[0]
    if T > max_frames:
        indices = torch.linspace(0, T - 1, max_frames).long()
        keypoints = keypoints[indices.numpy()]
    elif T < max_frames:
        pad = np.zeros((max_frames - T, 94, 4), dtype=np.float32)
        keypoints = np.concatenate([keypoints, pad], axis=0)

    keypoints = np.transpose(keypoints, (2, 1, 0))
    return keypoints, min(T, max_frames)


class PoseStitchDataset(Dataset):
    def __init__(self, split="train", max_frames=128, augment=None):
        csv_path = Path(cfg.POSE_STITCH_CSV_DIR) / f"isign_{split}_pose_stitch.csv"
        self.df = pd.read_csv(csv_path)
        self.augment = augment
        self.max_frames = max_frames
        print(f"  PoseStitch {split}: {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = row["uid"]
        text = row["text"]
        kp_path = KP_DIR / f"{uid}.npy"
        try:
            keypoints = np.load(kp_path).astype(np.float32)
        except Exception:
            keypoints = np.zeros((50, 94, 4), dtype=np.float32)

        if self.augment:
            keypoints = self.augment(keypoints)

        kp_tensor, n_frames = _normalize_keypoints(keypoints, self.max_frames)
        return {"keypoints": torch.from_numpy(kp_tensor).float(), "text": text,
                "video_name": uid, "num_frames": n_frames}


class SpaMoOFDataset(Dataset):
    def __init__(self, split="train", max_frames=128, augment=None):
        csv_path = Path(cfg.SPAMOOF_CSV_DIR) / f"{split}.csv"
        self.df = pd.read_csv(csv_path)
        self.df["text_ps"] = self.df["text"].apply(posestitch_preprocess)
        self.augment = augment
        self.max_frames = max_frames
        print(f"  SpaMo-OF {split}: {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_name = row["video_name"]
        text = row["text_ps"]
        kp_path = KP_DIR / f"{video_name}.npy"
        try:
            keypoints = np.load(kp_path).astype(np.float32)
        except Exception:
            keypoints = np.zeros((50, 94, 4), dtype=np.float32)

        if self.augment:
            keypoints = self.augment(keypoints)

        kp_tensor, n_frames = _normalize_keypoints(keypoints, self.max_frames)
        return {"keypoints": torch.from_numpy(kp_tensor).float(), "text": text,
                "video_name": video_name, "num_frames": n_frames}


class IndexedDataset(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        item["row_idx"] = idx
        return item


def collate_fn(batch):
    kp = torch.stack([b["keypoints"] for b in batch])
    row_idx = torch.tensor([b["row_idx"] for b in batch], dtype=torch.long)
    return {"keypoints": kp, "text": [b["text"] for b in batch],
            "num_frames": [b["num_frames"] for b in batch],
            "row_idx": row_idx}
