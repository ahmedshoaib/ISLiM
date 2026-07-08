"""
Keypoint Augmentation for Sign Language Videos

All spatial transforms apply uniformly across ALL frames to preserve temporal coherence.
All transforms are applied to the raw (T, 94, 4) keypoint sequence.

Augmentation pipeline (in order):
  1. horizontal_flip (50% probability)
  2. speed_perturbation (random factor)
  3. frame_dropout + interpolation
  4. normalize_keypoints() [done by caller]
  5. spatial_translate + spatial_scale [called together as spatial_augment]
  6. add_noise

Safe for sign language: horizontal flip remaps body parts correctly.
Unsafe (not implemented): rotation, vertical flip, per-frame transforms.
"""

import numpy as np
from typing import Tuple


class KeypointAugment:
    """
    Augmentation for sign language keypoint sequences (T, 94, 4).

    Keypoint layout:
      0:   LEFT_SHOULDER
      1:   RIGHT_SHOULDER
      2:   LEFT_ELBOW
      3:   RIGHT_ELBOW
      4:   LEFT_HIP
      5:   RIGHT_HIP
      6-26:  Left hand (21 pts - MediaPipe hand landmarks)
      27-47: Right hand (21 pts - MediaPipe hand landmarks)
      48-93: Face (46 pts - lips 20pts, eyes 16pts, eyebrows 10pts)

    Normalization: x,y,z -> [0,1] per sample. Augmentation operates on raw coords.
    """

    POSE_LEFT_RIGHT_SWAP = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}
    POSE_FIXED = set()

    LEFT_HAND_SLICE = slice(6, 27)
    RIGHT_HAND_SLICE = slice(27, 48)
    FACE_SLICE = slice(48, 94)

    LEFT_EYE = list(range(68, 76))
    RIGHT_EYE = list(range(76, 84))
    LEFT_EYEBROW = list(range(84, 89))
    RIGHT_EYEBROW = list(range(89, 94))
    MOUTH_CORNERS = [64, 65]

    def __init__(
        self,
        hflip_prob: float = 0.5,
        speed_factors: Tuple[float, ...] = (0.9, 1.0, 1.1),
        frame_dropout_rate: float = 0.05,
        translate_range: float = 0.02,
        scale_range: Tuple[float, float] = (0.95, 1.05),
        noise_std: float = 0.005,
        keypoint_dropout_rate: float = 0.02,
        enabled: bool = True,
    ):
        self.hflip_prob = hflip_prob
        self.speed_factors = list(speed_factors)
        self.frame_dropout_rate = frame_dropout_rate
        self.translate_range = translate_range
        self.scale_min, self.scale_max = scale_range
        self.noise_std = noise_std
        self.keypoint_dropout_rate = keypoint_dropout_rate
        self.enabled = enabled

    def __call__(
        self,
        keypoints: np.ndarray,
        augment_before_normalize: bool = True,
    ) -> np.ndarray:
        if not self.enabled:
            return keypoints

        kp = keypoints.astype(np.float32).copy()

        if augment_before_normalize:
            if np.random.rand() < self.hflip_prob:
                kp = self.horizontal_flip(kp)

            if self.speed_factors:
                factor = np.random.choice(self.speed_factors)
                if factor != 1.0:
                    kp = self.speed_perturb(kp, factor)

            if self.frame_dropout_rate > 0 and np.random.rand() < 0.5:
                kp = self.frame_dropout(kp, self.frame_dropout_rate)

        return kp

    def spatial_augment(self, keypoints: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return keypoints

        kp = keypoints.astype(np.float32).copy()

        if self.translate_range > 0:
            dx = np.random.uniform(-self.translate_range, self.translate_range)
            dy = np.random.uniform(-self.translate_range, self.translate_range)
            kp = self.spatial_translate(kp, dx, dy)

        if self.scale_min != 1.0 or self.scale_max != 1.0:
            scale = np.random.uniform(self.scale_min, self.scale_max)
            kp = self.spatial_scale(kp, scale)

        if self.keypoint_dropout_rate > 0 and np.random.rand() < 0.3:
            kp = self.keypoint_dropout(kp, self.keypoint_dropout_rate)

        if self.noise_std > 0:
            kp = self.add_noise(kp, self.noise_std)

        return kp

    def horizontal_flip(self, keypoints: np.ndarray) -> np.ndarray:
        kp = keypoints.copy()
        kp[:, :, 0] = 1.0 - kp[:, :, 0]

        left_hand = kp[:, self.LEFT_HAND_SLICE, :].copy()
        right_hand = kp[:, self.RIGHT_HAND_SLICE, :].copy()
        kp[:, self.LEFT_HAND_SLICE, :] = right_hand
        kp[:, self.RIGHT_HAND_SLICE, :] = left_hand

        for left_idx, right_idx in self.POSE_LEFT_RIGHT_SWAP.items():
            kp[:, [left_idx, right_idx], :] = kp[:, [right_idx, left_idx], :]

        left_eye = kp[:, self.LEFT_EYE, :].copy()
        right_eye = kp[:, self.RIGHT_EYE, :].copy()
        kp[:, self.LEFT_EYE, :] = right_eye
        kp[:, self.RIGHT_EYE, :] = left_eye

        left_eyebrow = kp[:, self.LEFT_EYEBROW, :].copy()
        right_eyebrow = kp[:, self.RIGHT_EYEBROW, :].copy()
        kp[:, self.LEFT_EYEBROW, :] = right_eyebrow
        kp[:, self.RIGHT_EYEBROW, :] = left_eyebrow

        kp[:, [64, 65], :] = kp[:, [65, 64], :]

        return kp

    def speed_perturb(self, keypoints: np.ndarray, factor: float) -> np.ndarray:
        T = keypoints.shape[0]
        new_T = max(5, int(round(T * factor)))
        indices = np.linspace(0, T - 1, new_T).astype(np.int64)
        return keypoints[indices]

    def frame_dropout(self, keypoints: np.ndarray, rate: float) -> np.ndarray:
        T = keypoints.shape[0]
        n_drop = max(1, int(round(T * rate)))

        keep_mask = np.ones(T, dtype=bool)
        drop_indices = np.random.choice(T, n_drop, replace=False)
        keep_mask[drop_indices] = False

        if not keep_mask[0]:
            keep_mask[0] = True
        if not keep_mask[-1]:
            keep_mask[-1] = True

        result = keypoints.copy()
        for t in range(T):
            if not keep_mask[t]:
                prev_vals = np.where(keep_mask[:t])[0]
                next_vals = np.where(keep_mask[t:])[0]
                if len(prev_vals) > 0 and len(next_vals) > 0:
                    prev_t = prev_vals[-1]
                    next_t = next_vals[0]
                    gap = next_t - prev_t
                    if gap > 1:
                        alpha = (t - prev_t) / gap
                        result[t] = (1 - alpha) * keypoints[prev_t] + alpha * keypoints[next_t]
                    elif gap == 1:
                        result[t] = keypoints[prev_t]
                    else:
                        result[t] = keypoints[prev_t]
                elif len(prev_vals) > 0:
                    result[t] = keypoints[prev_t]
                elif len(next_vals) > 0:
                    result[t] = keypoints[next_t]

        return result

    def spatial_translate(self, keypoints: np.ndarray, dx: float, dy: float) -> np.ndarray:
        result = keypoints.copy()
        result[:, :, 0] = np.clip(result[:, :, 0] + dx, 0.0, 1.0)
        result[:, :, 1] = np.clip(result[:, :, 1] + dy, 0.0, 1.0)
        return result

    def spatial_scale(self, keypoints: np.ndarray, scale: float) -> np.ndarray:
        result = keypoints.copy()
        visible = keypoints[:, :, 3] > 0.3
        valid_x = result[:, :, 0][visible]
        valid_y = result[:, :, 1][visible]

        if len(valid_x) == 0:
            return result

        center_x = np.mean(valid_x)
        center_y = np.mean(valid_y)
        result[:, :, 0] = np.clip(center_x + scale * (result[:, :, 0] - center_x), 0.0, 1.0)
        result[:, :, 1] = np.clip(center_y + scale * (result[:, :, 1] - center_y), 0.0, 1.0)
        return result

    def add_noise(self, keypoints: np.ndarray, std: float) -> np.ndarray:
        result = keypoints.copy()
        noise = np.random.normal(0.0, std, (result.shape[0], result.shape[1], 3)).astype(np.float32)
        result[:, :, :3] = np.clip(result[:, :, :3] + noise, 0.0, 1.0)
        return result

    def keypoint_dropout(self, keypoints: np.ndarray, rate: float) -> np.ndarray:
        result = keypoints.copy()
        T, K = result.shape[0], result.shape[1]
        n_drop = max(0, int(K * rate))
        if n_drop == 0:
            return result
        for t in range(T):
            drop_indices = np.random.choice(K, n_drop, replace=False)
            result[t, drop_indices, 3] = 0.0
            result[t, drop_indices, :3] = 0.0
        return result


class AugmentationConfig:
    CONSERVATIVE = {
        "hflip_prob": 0.5,
        "speed_factors": [1.0],
        "frame_dropout_rate": 0.0,
        "translate_range": 0.0,
        "scale_range": (1.0, 1.0),
        "noise_std": 0.0,
        "keypoint_dropout_rate": 0.0,
        "enabled": True,
    }

    MODERATE = {
        "hflip_prob": 0.5,
        "speed_factors": [0.9, 1.0, 1.1],
        "frame_dropout_rate": 0.05,
        "translate_range": 0.02,
        "scale_range": (0.95, 1.05),
        "noise_std": 0.005,
        "keypoint_dropout_rate": 0.02,
        "enabled": True,
    }

    AGGRESSIVE = {
        "hflip_prob": 0.5,
        "speed_factors": [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15],
        "frame_dropout_rate": 0.1,
        "translate_range": 0.03,
        "scale_range": (0.92, 1.08),
        "noise_std": 0.01,
        "keypoint_dropout_rate": 0.05,
        "enabled": True,
    }

    @classmethod
    def get(cls, name: str = "MODERATE") -> dict:
        return getattr(cls, name, cls.MODERATE).copy()
