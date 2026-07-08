#!/usr/bin/env python3
"""
Extract ISLiM's 94-keypoint structure from iSign videos using MediaPipe Holistic.

Keypoint layout (94 total, 4 values each: x, y, z, visibility):
    0–5:    Body (6: L/R shoulder, L/R elbow, L/R hip)
    6–26:   Left hand (21: wrist + finger joints)
    27–47:  Right hand (21, same structure)
    48–67:  Lips (20 landmarks)
    68–83:  Eyes (16 landmarks)
    84–93:  Eyebrows (10 landmarks)

Output: .npy files with shape (num_frames, 94, 4) as float16.
Supports multiprocessing, chunk-based resumption, and auto-skip of
existing valid files.

Usage:
    python scripts/preprocess/extract_keypoints.py
    python scripts/preprocess/extract_keypoints.py --workers 8 --chunk_size 50
"""

import os, sys, re, argparse, warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config as cfg

# ── MediaPipe indices ──
mp_holistic = mp.solutions.holistic

POSE_INDICES = [
    mp_holistic.PoseLandmark.LEFT_SHOULDER,
    mp_holistic.PoseLandmark.RIGHT_SHOULDER,
    mp_holistic.PoseLandmark.LEFT_ELBOW,
    mp_holistic.PoseLandmark.RIGHT_ELBOW,
    mp_holistic.PoseLandmark.LEFT_HIP,
    mp_holistic.PoseLandmark.RIGHT_HIP,
]
FACE_LIPS = [61,146,91,181,84,17,314,405,321,375,291,409,270,95,88,178,87,14,78,308]
FACE_EYES = [33,133,160,158,153,144,145,246,362,263,387,385,380,373,374,249]
FACE_EYEBROWS = [107,66,105,63,70,336,296,334,293,300]
FACE_INDICES = FACE_LIPS + FACE_EYES + FACE_EYEBROWS

NUM_KEYPOINTS = 94
KEYPOINT_COORDS = 4
MAX_MISSING_CONSECUTIVE = 5


def create_holistic():
    return mp_holistic.Holistic(
        static_image_mode=False, model_complexity=1,
        smooth_landmarks=True, refine_face_landmarks=False,
        min_tracking_confidence=0.5, min_detection_confidence=0.5)


def extract_frame_keypoints(frame, holistic_model):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = holistic_model.process(rgb)
    kp = np.full((NUM_KEYPOINTS, KEYPOINT_COORDS), 0.0, dtype=np.float32)

    if res.pose_landmarks:
        for i, idx in enumerate(POSE_INDICES):
            lm = res.pose_landmarks.landmark[idx]
            kp[i] = [lm.x, lm.y, lm.z, lm.visibility]

    if res.left_hand_landmarks:
        for i in range(21):
            lm = res.left_hand_landmarks.landmark[i]
            kp[6 + i] = [lm.x, lm.y, lm.z, 1.0]

    if res.right_hand_landmarks:
        for i in range(21):
            lm = res.right_hand_landmarks.landmark[i]
            kp[27 + i] = [lm.x, lm.y, lm.z, 1.0]

    if res.face_landmarks:
        for i, idx in enumerate(FACE_INDICES):
            lm = res.face_landmarks.landmark[idx]
            kp[48 + i] = [lm.x, lm.y, lm.z, 1.0 if lm.z > -0.5 else 0.5]

    return kp


def interpolate_keypoints(kp_seq):
    """Linear interpolation over gaps <= MAX_MISSING_CONSECUTIVE frames."""
    T, K, C = kp_seq.shape
    for k in range(K):
        valid = np.where(kp_seq[:, k, 3] > 0.5)[0]
        if len(valid) == 0:
            kp_seq[:, k, :] = 0.0; continue
        for i in range(len(valid)-1):
            a, b = valid[i], valid[i+1]
            gap = b - a
            if 1 < gap <= MAX_MISSING_CONSECUTIVE:
                for j in range(1, gap):
                    alpha = j / gap
                    kp_seq[a+j, k, :3] = (1-alpha)*kp_seq[a,k,:3] + alpha*kp_seq[b,k,:3]
                    kp_seq[a+j, k, 3] = (1-alpha)*kp_seq[a,k,3] + alpha*kp_seq[b,k,3]
        fv, lv = valid[0], valid[-1]
        for f in range(fv):
            if kp_seq[f, k, 3] < 0.5: kp_seq[f, k] = kp_seq[fv, k]
        for f in range(lv+1, T):
            if kp_seq[f, k, 3] < 0.5: kp_seq[f, k] = kp_seq[lv, k]
    kp_seq[kp_seq[:,:,3] < 0.5] = 0.0
    return kp_seq


def process_video(video_path, output_dir):
    """Extract keypoints for one video. Returns (path, status, message)."""
    uid = Path(video_path).stem
    out_path = os.path.join(output_dir, f"{uid}.npy")

    if os.path.exists(out_path):
        try:
            if np.load(out_path).shape[1] == NUM_KEYPOINTS:
                return video_path, "skipped", ""
        except: pass

    holistic = create_holistic()
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release(); holistic.close()
            return video_path, "error", "cannot_open"
        frames_kp = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frames_kp.append(extract_frame_keypoints(frame, holistic))
        cap.release(); holistic.close()
        if not frames_kp: return video_path, "error", "no_frames"
        kp_seq = np.stack(frames_kp, axis=0)
        kp_seq = interpolate_keypoints(kp_seq)
        kp_seq = kp_seq.astype(np.float16)
        np.save(out_path, kp_seq)
        return video_path, "ok", f"{kp_seq.shape[0]} frames"
    except Exception as e:
        return video_path, "error", str(e)


def find_videos(video_dir):
    """Find MP4 files (flat or nested)."""
    exts = {".mp4", ".mov", ".avi", ".MP4", ".MOV", ".AVI"}
    return sorted([str(p) for p in Path(video_dir).rglob("*")
                   if p.suffix in exts])


def main():
    parser = argparse.ArgumentParser(description="iSign 94-keypoint extraction")
    parser.add_argument("--input", default=str(cfg.ISIGN_VIDEOS_DIR),
                        help="Video directory")
    parser.add_argument("--output", default=str(cfg.ISIGN_KEYPOINTS_DIR),
                        help="Output directory for .npy files")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=100,
                        help="Videos to process before GC pause")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of videos")
    args = parser.parse_args()

    videos = find_videos(args.input)
    if args.limit: videos = videos[:args.limit]

    print(f"Found {len(videos)} videos")
    print(f"Output: {args.output}")
    print(f"Workers: {args.workers}, chunk size: {args.chunk_size}")

    os.makedirs(args.output, exist_ok=True)

    ok, skip, err = 0, 0, 0
    for start in range(0, len(videos), args.chunk_size):
        chunk = videos[start:start+args.chunk_size]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process_video, v, args.output): v for v in chunk}
            for f in tqdm(as_completed(futures), total=len(chunk),
                          desc=f"Chunk {start//args.chunk_size+1}"):
                path, status, msg = f.result()
                if status == "ok": ok += 1
                elif status == "skipped": skip += 1
                else:
                    err += 1
                    if err <= 5: print(f"  {msg}: {path}")

    print(f"\nDone. OK={ok}  Skipped={skip}  Errors={err}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
