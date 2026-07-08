#!/usr/bin/env python3
"""
Compare keypoints: 2D overlay vs 3D region split (body / hands / face).

For one frame per video:
  - Left:    2D overlay on video frame (all 94 pts: body + hands + face)
  - Middle:  3D frontal - body | hands | face side by side
  - Right:   3D - hands (2 profiles: side + top-down) | face (bottom-up)

Keypoint layout (94 landmarks):
    0-5:    Body (6: L/R shoulder, L/R elbow, L/R hip)
    6-26:   Left hand (21)
    27-47:  Right hand (21)
    48-93:  Face (46: lips 20, eyes 16, eyebrows 10)

Usage:
    python scripts/viz/keypoint_comparison.py
"""

import os
import sys
import random
import math
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config import ISIGN_KEYPOINTS_DIR, ISIGN_VIDEOS_DIR, POSE_STITCH_CSV_DIR
from pathlib import Path

OUTPUT_DIR = _PROJECT_ROOT / "viz" / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VIDEO_DIR = str(ISIGN_VIDEOS_DIR)
KP_DIR = str(ISIGN_KEYPOINTS_DIR)
TEST_CSV = str(Path(POSE_STITCH_CSV_DIR) / "isign_test_pose_stitch.csv")

random.seed(42)

POSE_IDX = list(range(0, 6))
LEFT_HAND_IDX = list(range(6, 27))
RIGHT_HAND_IDX = list(range(27, 48))
FACE_IDX = list(range(48, 94))

BODY_CONNECTIONS = [(0, 2), (0, 4), (1, 3), (1, 5), (0, 1)]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

FACE_CONNS_LOCAL = sum([
    [(i, i + 1) for i in range(0, 19)] + [(19, 0)],
    [(i, i + 1) for i in range(20, 27)] + [(27, 20)],
    [(i, i + 1) for i in range(28, 35)] + [(35, 28)],
    [(i, i + 1) for i in range(36, 40)],
    [(i, i + 1) for i in range(41, 45)],
], [])

FACE_SUB_LIPS = list(range(0, 20))
FACE_SUB_LEYE = list(range(20, 28))
FACE_SUB_REYE = list(range(28, 36))
FACE_SUB_LBROW = list(range(36, 41))
FACE_SUB_RBROW = list(range(41, 46))


def _face_conns_local(idxs):
    return [(i, j) for i, j in FACE_CONNS_LOCAL if i in idxs and j in idxs]


FACE_CONNS_LIPS = _face_conns_local(FACE_SUB_LIPS)
FACE_CONNS_LEYE = _face_conns_local(FACE_SUB_LEYE)
FACE_CONNS_REYE = _face_conns_local(FACE_SUB_REYE)
FACE_CONNS_LBROW = _face_conns_local(FACE_SUB_LBROW)
FACE_CONNS_RBROW = _face_conns_local(FACE_SUB_RBROW)

C_BODY = "#2196F3"
C_LHAND = "#4CAF50"
C_RHAND = "#F44336"
C_FACE = "#FF9800"
C_BG = "#1a1a2e"
C_GRID = "#333355"
C_MOUTH = "#FF5252"
C_EYES = "#4FC3F7"
C_BROWS = "#CE93D8"


def get_uids(n=12):
    uids = []
    with open(TEST_CSV) as f:
        for line in f:
            uid = line.strip().split(",")[0]
            if uid and uid != "uid":
                uids.append(uid)
    return random.sample(uids, min(n, len(uids)))


def load_video(uid):
    path = os.path.join(VIDEO_DIR, f"{uid}.mp4")
    if not os.path.exists(path):
        return None, 0
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, total


def load_keypoints(uid):
    path = os.path.join(KP_DIR, f"{uid}.npy")
    return np.load(path) if os.path.exists(path) else None


def hand_conns(offset):
    return [(i + offset, j + offset) for i, j in HAND_CONNECTIONS]


def face_sub_conns(conns, offset=48):
    return [(i + offset, j + offset) for i, j in conns]


def normalize_region(kp, lo, hi):
    seg = kp[lo:hi + 1].copy()
    valid = seg[:, 3] > 0.3
    if valid.sum() < 2:
        return seg
    for dim in range(3):
        vals = seg[:, dim]
        lo_v, hi_v = vals[valid].min(), vals[valid].max()
        if hi_v > lo_v:
            seg[:, dim] = (vals - lo_v) / (hi_v - lo_v)
        else:
            seg[:, dim] = 0.5
    return seg


def draw_2d(ax, frame, kp, title):
    h, w = frame.shape[:2]
    ax.imshow(frame)
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", color="#111", pad=6)

    for i, j in BODY_CONNECTIONS:
        if kp[i, 3] > 0.3 and kp[j, 3] > 0.3:
            ax.plot([kp[i, 0] * w, kp[j, 0] * w], [kp[i, 1] * h, kp[j, 1] * h],
                    color=C_BODY, lw=1.5, alpha=0.7)

    for offset_, c in ((0, C_LHAND), (21, C_RHAND)):
        for i, j in hand_conns(offset_):
            ii, jj = i + 6, j + 6
            if kp[ii, 3] > 0.3 and kp[jj, 3] > 0.3:
                ax.plot([kp[ii, 0] * w, kp[jj, 0] * w], [kp[ii, 1] * h, kp[jj, 1] * h],
                        color=c, lw=1, alpha=0.6)

    for conns, c in [
        (face_sub_conns(FACE_CONNS_LIPS, 48), C_MOUTH),
        (face_sub_conns(FACE_CONNS_LEYE, 48), C_EYES),
        (face_sub_conns(FACE_CONNS_REYE, 48), C_EYES),
        (face_sub_conns(FACE_CONNS_LBROW, 48), C_BROWS),
        (face_sub_conns(FACE_CONNS_RBROW, 48), C_BROWS),
    ]:
        for i, j in conns:
            if kp[i, 3] > 0.3 and kp[j, 3] > 0.3:
                ax.plot([kp[i, 0] * w, kp[j, 0] * w], [kp[i, 1] * h, kp[j, 1] * h],
                        color=c, lw=0.8, alpha=0.4)

    for idx in range(94):
        if kp[idx, 3] <= 0.3:
            continue
        if idx in POSE_IDX:
            c, s = C_BODY, 32
        elif idx in LEFT_HAND_IDX:
            c, s = C_LHAND, 14
        elif idx in RIGHT_HAND_IDX:
            c, s = C_RHAND, 14
        elif idx <= 67:
            c, s = C_MOUTH, 10
        elif idx <= 83:
            c, s = C_EYES, 10
        else:
            c, s = C_BROWS, 8
        ax.scatter(kp[idx, 0] * w, kp[idx, 1] * h, color=c, s=s,
                   edgecolors="white", linewidth=0.3, alpha=0.85, zorder=5)


def draw_3d_region(ax, kp, region, elev, azim, title):
    ax.set_facecolor(C_BG)
    ax.set_title(title, fontsize=10, fontweight="bold", color="white", pad=4)

    if region == "body":
        lo, hi = 0, 5
        idxs = POSE_IDX
        s = 50
    elif region == "hands":
        lo, hi = 6, 47
        s = 18
    else:
        lo, hi = 48, 93

    kp_r = normalize_region(kp, lo, hi)
    x, y, z = kp_r[:, 0], kp_r[:, 1], kp_r[:, 2]

    if region == "body":
        for i, j in BODY_CONNECTIONS:
            if kp[i, 3] > 0.3 and kp[j, 3] > 0.3:
                ax.plot([x[i - lo], x[j - lo]], [y[i - lo], y[j - lo]], [z[i - lo], z[j - lo]],
                        color=C_BODY, lw=2, alpha=0.6)
    elif region == "hands":
        for offset_, c_h in ((0, C_LHAND), (21, C_RHAND)):
            for i, j in hand_conns(offset_):
                ii, jj = i + 6, j + 6
                if kp[ii, 3] > 0.3 and kp[jj, 3] > 0.3:
                    ax.plot([x[ii - lo], x[jj - lo]], [y[ii - lo], y[jj - lo]], [z[ii - lo], z[jj - lo]],
                            color=c_h, lw=1.5, alpha=0.5)
        for ii in LEFT_HAND_IDX:
            if kp[ii, 3] > 0.3:
                ax.scatter(x[ii - lo], y[ii - lo], z[ii - lo], color=C_LHAND, s=s,
                           alpha=0.85, edgecolors="white", linewidth=0.3, zorder=5)
        for ii in RIGHT_HAND_IDX:
            if kp[ii, 3] > 0.3:
                ax.scatter(x[ii - lo], y[ii - lo], z[ii - lo], color=C_RHAND, s=s,
                           alpha=0.85, edgecolors="white", linewidth=0.3, zorder=5)
    else:
        subs = [
            (48, 67, C_MOUTH, face_sub_conns(FACE_CONNS_LIPS, 48), 20),
            (68, 75, C_EYES, face_sub_conns(FACE_CONNS_LEYE, 48), 20),
            (76, 83, C_EYES, face_sub_conns(FACE_CONNS_REYE, 48), 20),
            (84, 88, C_BROWS, face_sub_conns(FACE_CONNS_LBROW, 48), 14),
            (89, 93, C_BROWS, face_sub_conns(FACE_CONNS_RBROW, 48), 14),
        ]
        for s_lo, s_hi, c_sub, conns, sz in subs:
            for i, j in conns:
                if kp[i, 3] > 0.3 and kp[j, 3] > 0.3:
                    ax.plot([x[i - lo], x[j - lo]], [y[i - lo], y[j - lo]], [z[i - lo], z[j - lo]],
                            color=c_sub, lw=0.8, alpha=0.3)
            for ii in range(s_lo, s_hi + 1):
                if kp[ii, 3] > 0.3:
                    ax.scatter(x[ii - lo], y[ii - lo], z[ii - lo], color=c_sub, s=sz,
                               alpha=0.85, edgecolors="white", linewidth=0.3, zorder=5)

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_zlim(0, 1)
    ax.grid(False)
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=elev, azim=azim)


def make_video_figure(uid, frame_idx=None):
    kp_all = load_keypoints(uid)
    frames, total_frames = load_video(uid)
    if kp_all is None or not frames:
        print(f"  Skip {uid}: no data")
        return None

    if frame_idx is None:
        mid = min(total_frames, len(kp_all)) // 2
        idx = max(5, min(mid, len(kp_all) - 1, total_frames - 1))
    else:
        idx = min(frame_idx, len(kp_all) - 1, total_frames - 1)
    frame = frames[idx]
    kp = kp_all[idx]

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("white")

    gs = fig.add_gridspec(3, 3, hspace=0.25, wspace=0.22,
                           left=0.06, right=0.97, top=0.92, bottom=0.07)
    fig.suptitle(f"{uid} - Frame {idx + 1}/{total_frames}",
                 fontsize=13, fontweight="bold", color="#333", y=0.96)

    ax0 = fig.add_subplot(gs[:, 0])
    draw_2d(ax0, frame, kp, "2D Overlay\n(all 94 pts)")

    draw_3d_region(fig.add_subplot(gs[0, 1], projection="3d"),
                   kp, "body", elev=0, azim=-90, title="Body (frontal)")
    draw_3d_region(fig.add_subplot(gs[1, 1], projection="3d"),
                   kp, "hands", elev=0, azim=-90, title="Hands (frontal)")
    draw_3d_region(fig.add_subplot(gs[2, 1], projection="3d"),
                   kp, "face", elev=0, azim=-90, title="Face (frontal)")

    draw_3d_region(fig.add_subplot(gs[0, 2], projection="3d"),
                   kp, "hands", elev=0, azim=0, title="Hands (profile)")
    draw_3d_region(fig.add_subplot(gs[1, 2], projection="3d"),
                   kp, "hands", elev=90, azim=0, title="Hands (top-down)")
    draw_3d_region(fig.add_subplot(gs[2, 2], projection="3d"),
                   kp, "face", elev=-60, azim=0, title="Face (bottom-up)")

    from matplotlib.lines import Line2D
    fig.legend(
        handles=[
            Line2D([], [], color=C_BODY, marker="o", linestyle="", label="Body (6)"),
            Line2D([], [], color=C_LHAND, marker="o", linestyle="", label="L Hand (21)"),
            Line2D([], [], color=C_RHAND, marker="o", linestyle="", label="R Hand (21)"),
            Line2D([], [], color=C_MOUTH, marker="o", linestyle="", label="Mouth (20)"),
            Line2D([], [], color=C_EYES, marker="o", linestyle="", label="Eyes (16)"),
            Line2D([], [], color=C_BROWS, marker="o", linestyle="", label="Brows (10)"),
        ],
        loc="lower center", ncol=6, fontsize=9,
        framealpha=0.0, handletextpad=0.6, markerscale=0.7,
    )

    if frame_idx is not None:
        out = os.path.join(OUTPUT_DIR, f"video_{uid}_f{idx:04d}.png")
    else:
        out = os.path.join(OUTPUT_DIR, f"video_{uid}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out}")
    return out


def make_collage():
    import glob
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "video_*.png")))
    if not files:
        return
    images = []
    for f in files:
        im = cv2.imread(f)
        if im is not None:
            images.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

    if not images:
        return
    n = len(images)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    h = max(im.shape[0] for im in images)
    w = max(im.shape[1] for im in images)
    canvas = np.ones((rows * h, cols * w, 3), dtype=np.uint8) * 255
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        ih, iw = img.shape[:2]
        y_off = (h - ih) // 2
        x_off = (w - iw) // 2
        canvas[r * h + y_off:r * h + y_off + ih, c * w + x_off:c * w + x_off + iw] = img
    out = os.path.join(OUTPUT_DIR, "collage.png")
    plt.figure(figsize=(cols * 6, rows * 4))
    plt.imshow(canvas)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out}")


def main():
    target_uid = "NwVF-WWZ0Bc--48"
    kp_all = load_keypoints(target_uid)
    if kp_all is None:
        print(f"No keypoints for {target_uid}")
        return
    total = kp_all.shape[0]
    frame_idxs = list(range(total))

    print(f"=== Focus: {target_uid} - all {total} frames ===\n")
    print(f"=== Focus: {target_uid} - frames {frame_idxs[0]}-{frame_idxs[-1]} ===\n")

    out_paths = []
    for i, idx in enumerate(frame_idxs):
        print(f"[{i + 1}/{len(frame_idxs)}] frame {idx}")
        path = make_video_figure(target_uid, frame_idx=idx)
        if path:
            out_paths.append(path)

    print(f"\nDone. {len(out_paths)} figures in {OUTPUT_DIR}/")
    print("\nCreating collage...")
    make_collage()
    print(f"\nDone. Output in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
