import os
import cv2
import csv
import numpy as np
from collections import namedtuple

# ── CONFIG ────────────────────────────────────────────────────────────────
# Point this to your BIWI‐custom root “faces_0” folder
DATA_ROOT = r"D:\archive\faces_0"
OUT_CSV   = "biwi_custom_eval.csv"

# reuse your existing head‐pose estimator
from multi_pipeline_gaze import get_head_pose, mp_face_mesh

FramePose = namedtuple("FramePose", ["pitch", "yaw"])


def load_pose_txt(fn):
    """
    Read the 4th non‐blank line of frame_xxxx_pose.txt,
    which is already "pitch yaw depth".
    """
    lines = [l.strip() for l in open(fn, "r") if l.strip()]
    # line 0,1,2 = rotation matrix; line 3 = "<pitch> <yaw> <depth>"
    pitch, yaw, _ = map(float, lines[3].split())
    return FramePose(pitch, yaw)


def load_rgb_image(pose_txt_path):
    """
    Given the full path to "frame_XXXX_pose.txt",
    load "frame_XXXX_rgb.png" from the same folder.
    """
    rgb_path = pose_txt_path.replace("_pose.txt", "_rgb.png")
    img = cv2.imread(rgb_path)
    if img is None:
        raise FileNotFoundError(f"Couldn't load RGB at {rgb_path}")
    return img


def detect_landmarks_largest(img):
    """
    Run MediaPipe in static mode on a single image,
    return the landmarks of the largest face (or None).
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = mp_face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None

    # pick the face with the largest bounding‐box area
    def bbox_area(fm):
        xs = [p.x for p in fm.landmark]
        ys = [p.y for p in fm.landmark]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))

    best = max(res.multi_face_landmarks, key=bbox_area)
    return best.landmark


if __name__ == "__main__":
    # open CSV
    with open(OUT_CSV, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow([
            "subject", "frame",
            "gt_pitch", "gt_yaw",
            "est_pitch", "est_yaw",
            "err_pitch", "err_yaw"
        ])

        # iterate subjects (folders "01".."24")
        for subj in sorted(os.listdir(DATA_ROOT)):
            subj_dir = os.path.join(DATA_ROOT, subj)
            # find all pose files
            pose_files = sorted(f for f in os.listdir(subj_dir) if f.endswith("_pose.txt"))

            for pose_fn in pose_files:
                frame_id    = pose_fn.split("_")[1].split(".")[0]
                pose_path   = os.path.join(subj_dir, pose_fn)

                # 1) load GT pitch,yaw
                gt = load_pose_txt(pose_path)

                # 2) load the RGB PNG
                try:
                    img = load_rgb_image(pose_path)
                except FileNotFoundError as e:
                    print(f"⚠️  Skipping {subj} frame {frame_id}: {e}")
                    continue

                # 3) detect face & landmarks
                lms = detect_landmarks_largest(img)
                if lms is None:
                    print(f"⚠️  No face in {subj} frame {frame_id}")
                    continue

                # 4) estimate pitch,yaw
                est_p, est_y = get_head_pose(lms, img.shape[:2])
                if est_p is None:
                    print(f"⚠️  PnP failed in {subj} frame {frame_id}")
                    continue

                # 5) compute errors & write
                err_p = abs(est_p - gt.pitch)
                err_y = abs(est_y - gt.yaw)
                writer.writerow([
                    subj, frame_id,
                    f"{gt.pitch:.2f}", f"{gt.yaw:.2f}",
                    f"{est_p:.2f}",   f"{est_y:.2f}",
                    f"{err_p:.2f}",   f"{err_y:.2f}"
                ])

    print(f"✅ BIWI evaluation complete → {OUT_CSV}")
