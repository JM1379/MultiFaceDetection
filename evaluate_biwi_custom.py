# evaluate_biwi_custom.py

import os
import cv2
import csv
import numpy as np
from collections import namedtuple

# ── ADJUST THESE ────────────────────────────────────────────────────────────
DATA_ROOT = r"D:\archive\faces_0"      # path to your faces_0 folder
OUT_CSV   = "biwi_custom_eval.csv"     # output CSV
# ─────────────────────────────────────────────────────────────────────────────

FramePose = namedtuple("FramePose", ["pitch","yaw"])

# ── Copy your model‐points & landmark indices here ──────────────────────────
MODEL_POINTS = np.array([
    (0.0,   0.0,    0.0),    # nose tip
    (0.0, -330.0, -65.0),    # chin
    (-225.0,170.0,-135.0),   # left eye outer corner
    (225.0,170.0,-135.0),    # right eye outer corner
    (-150.0,-150.0,-125.0),  # left mouth corner
    (150.0,-150.0,-125.0)    # right mouth corner
], dtype="double")

LM_IDX = {
    "nose":       1,
    "chin":      152,
    "left_eye":   33,
    "right_eye":263,
    "left_mouth": 61,
    "right_mouth":291,
}
# ─────────────────────────────────────────────────────────────────────────────

def rotation_matrix_to_euler(R):
    """
    Decompose 3×3 rotation matrix R into Euler angles via RQ.
    Returns (rx, ry) in degrees, where:
      rx = rotation about X axis ("pitch"), 
      ry = rotation about Y axis ("yaw").
    """
    # cv2.RQDecomp3x3 returns: retval, mtxR, mtxQ, xRot, yRot, zRot 
    _, _, _, rx, ry, _ = cv2.RQDecomp3x3(R)
    return rx, ry

def load_pose_txt(path):
    """
    Reads the first 3 lines of frame_XXXX_pose.txt as R,
    then returns FramePose(pitch, yaw) via RQ decomposition.
    """
    lines = [l.strip() for l in open(path, "r") if l.strip()]
    Rmat = np.array([list(map(float, lines[i].split())) for i in range(3)])
    pitch, yaw = rotation_matrix_to_euler(Rmat)
    return FramePose(pitch, yaw)

def load_rgb_image(pose_txt_path):
    """
    Convert '…_pose.txt' → '…_rgb.png' and load that image.
    """
    rgb_path = pose_txt_path.replace("_pose.txt", "_rgb.png")
    img = cv2.imread(rgb_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load RGB at {rgb_path}")
    return img

def detect_landmarks_largest(img):
    """
    Run MediaPipe in static mode (one face max), return landmarks list.
    """
    import mediapipe as mp
    mpfm = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5
    )
    res = mpfm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    mpfm.close()
    if not res.multi_face_landmarks:
        return None
    # we only requested max_num_faces=1
    return res.multi_face_landmarks[0].landmark

def estimate_head_pose(landmarks, frame_size):
    """
    SolvePnP → Rodrigues → R → Euler via RQDecomp3x3.
    Returns (pitch, yaw) in degrees.
    """
    h, w = frame_size
    img_pts = np.array([
        (landmarks[LM_IDX["nose"]].x  * w,
         landmarks[LM_IDX["nose"]].y  * h),
        (landmarks[LM_IDX["chin"]].x  * w,
         landmarks[LM_IDX["chin"]].y  * h),
        (landmarks[LM_IDX["left_eye"]].x * w,
         landmarks[LM_IDX["left_eye"]].y * h),
        (landmarks[LM_IDX["right_eye"]].x * w,
         landmarks[LM_IDX["right_eye"]].y * h),
        (landmarks[LM_IDX["left_mouth"]].x * w,
         landmarks[LM_IDX["left_mouth"]].y * h),
        (landmarks[LM_IDX["right_mouth"]].x* w,
         landmarks[LM_IDX["right_mouth"]].y* h)
    ], dtype="double")

    focal = w
    center = (w/2, h/2)
    cam_mat = np.array([[focal, 0, center[0]],
                        [0, focal, center[1]],
                        [0,     0,          1]], dtype="double")
    dist_coeffs = np.zeros((4,1))

    ok, rvec, _ = cv2.solvePnP(
        MODEL_POINTS, img_pts,
        cam_mat, dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP
    )
    if not ok:
        return None, None

    R_est, _ = cv2.Rodrigues(rvec)
    return rotation_matrix_to_euler(R_est)

if __name__ == "__main__":
    with open(OUT_CSV, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow([
            "subject","frame",
            "gt_pitch","gt_yaw",
            "est_pitch","est_yaw",
            "err_pitch","err_yaw"
        ])

        for subj in sorted(os.listdir(DATA_ROOT)):
            subj_dir = os.path.join(DATA_ROOT, subj)
            pose_files = sorted(f for f in os.listdir(subj_dir) if f.endswith("_pose.txt"))

            for pose_fn in pose_files:
                frame_id  = pose_fn.split("_")[1].split(".")[0]
                pose_path = os.path.join(subj_dir, pose_fn)

                # 1) ground‐truth
                gt = load_pose_txt(pose_path)

                # 2) image
                try:
                    img = load_rgb_image(pose_path)
                except FileNotFoundError as e:
                    print(f"⚠️ Skipping {subj}#{frame_id}: {e}")
                    continue

                # 3) landmarks
                lms = detect_landmarks_largest(img)
                if lms is None:
                    print(f"⚠️ No face in {subj}#{frame_id}")
                    continue

                # 4) estimate
                est_p, est_y = estimate_head_pose(lms, img.shape[:2])
                if est_p is None:
                    print(f"⚠️ PnP failed {subj}#{frame_id}")
                    continue

                # 5) log
                err_p = abs(est_p - gt.pitch)
                err_y = abs(est_y - gt.yaw)
                writer.writerow([
                    subj, frame_id,
                    f"{gt.pitch:.2f}", f"{gt.yaw:.2f}",
                    f"{est_p:.2f}",   f"{est_y:.2f}",
                    f"{err_p:.2f}",   f"{err_y:.2f}"
                ])

    print("✅ BIWI evaluation complete →", OUT_CSV)
