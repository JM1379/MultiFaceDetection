import cv2
import numpy as np
import mediapipe as mp
import sys
from collections import deque, defaultdict

sys.path.append(r'C:\Users\Julian\vis_impair_projects\sort')
from sort import Sort

# ── CONFIG ────────────────────────────────────────────────────────────────
MAX_FACES     = 3
YAW_THRESH    = 39.0    # degrees tolerance for yaw
PITCH_THRESH  = 8.0     # degrees tolerance for pitch
HISTORY_LEN   = 5       # frames to consider for smoothing
VOTE_THRESHOLD= 3       # required True votes in HISTORY_LEN
CONSEC_FRAMES = 30      # consecutive frames of stable look to trigger event

# 3D model points for PnP
MODEL_POINTS = np.array([
    (0.0,   0.0,    0.0),    # nose tip
    (0.0, -330.0, -65.0),    # chin
    (-225.0,170.0,-135.0),    # left eye outer corner
    (225.0,170.0,-135.0),     # right eye outer corner
    (-150.0,-150.0,-125.0),   # left mouth corner
    (150.0,-150.0,-125.0)     # right mouth corner
], dtype='double')

# Landmark indices
LM_IDX = {
    'nose': 1,
    'chin': 152,
    'left_eye': 33,
    'right_eye': 263,
    'left_mouth': 61,
    'right_mouth': 291
}

# initialise MediaPipe Face Mesh
tmp = mp.solutions.face_mesh.FaceMesh
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=MAX_FACES,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


def get_head_pose(landmarks, frame_size):
    """Estimate (pitch, yaw) in degrees."""
    h, w = frame_size
    # 2D image points
    pts = np.array([
        (landmarks[LM_IDX['nose']].x * w, landmarks[LM_IDX['nose']].y * h),
        (landmarks[LM_IDX['chin']].x * w, landmarks[LM_IDX['chin']].y * h),
        (landmarks[LM_IDX['left_eye']].x * w, landmarks[LM_IDX['left_eye']].y * h),
        (landmarks[LM_IDX['right_eye']].x * w, landmarks[LM_IDX['right_eye']].y * h),
        (landmarks[LM_IDX['left_mouth']].x * w, landmarks[LM_IDX['left_mouth']].y * h),
        (landmarks[LM_IDX['right_mouth']].x * w, landmarks[LM_IDX['right_mouth']].y * h)
    ], dtype='double')

    # camera params
    focal = w
    center = (w/2, h/2)
    cam_mat = np.array([
        [focal, 0, center[0]],
        [0, focal, center[1]],
        [0,     0,        1]
    ], dtype='double')
    dist_coeffs = np.zeros((4,1))

    success, rvec, _ = cv2.solvePnP(
        MODEL_POINTS, pts, cam_mat, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None, None

    R, _ = cv2.Rodrigues(rvec)
    # compute Euler angles
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    pitch = np.degrees(np.arctan2(R[2,1], R[2,2]))
    yaw   = np.degrees(np.arctan2(-R[2,0], sy))

    # normalize pitch to [-180,180]
    if pitch > 180:
        pitch -= 360
    elif pitch < -180:
        pitch += 360
    # map pitch into [-90,90]
    if pitch > 90:
        pitch = 180 - pitch
    elif pitch < -90:
        pitch = -180 - pitch

    return pitch, yaw


def main(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Opened {video_path}: {total} frames @ {fps:.1f}fps")

    delay = max(1, int(1000/fps))
    recorder = Sort(max_age=30, min_hits=1, iou_threshold=0.3)

    histories    = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
    consec_count = defaultdict(int)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video ended")
            break
        frame_idx += 1

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(rgb)

        # build detections & landmarks
        dets, landmarks_list = [], []
        if results.multi_face_landmarks:
            for lm_set in results.multi_face_landmarks:
                xs = [lm.x for lm in lm_set.landmark]
                ys = [lm.y for lm in lm_set.landmark]
                x1, x2 = int(min(xs)*w), int(max(xs)*w)
                y1, y2 = int(min(ys)*h), int(max(ys)*h)
                dets.append([x1, y1, x2, y2, 1.0])
                landmarks_list.append(lm_set.landmark)

        # update tracks
        tracks = recorder.update(np.array(dets)) if dets else []

        for x1, y1, x2, y2, tid in tracks:
            tid = int(tid)
            # find associated landmarks by nearest center
            cx, cy = (x1+x2)/2, (y1+y2)/2
            best_i, best_dist = None, float('inf')
            for i, d in enumerate(dets):
                dcx, dcy = (d[0]+d[2])/2, (d[1]+d[3])/2
                dist = (dcx-cx)**2 + (dcy-cy)**2
                if dist < best_dist:
                    best_dist, best_i = dist, i

            lm = landmarks_list[best_i]
            pitch, yaw = get_head_pose(lm, (h, w))
            if pitch is None:
                continue

            # static threshold check
            is_looking = (abs(yaw) < YAW_THRESH) and (abs(pitch) < PITCH_THRESH)

            # smoothing & voting
            hist = histories[tid]
            hist.append(is_looking)
            stable = sum(hist) >= VOTE_THRESHOLD

            if stable:
                consec_count[tid] += 1
                if consec_count[tid] == CONSEC_FRAMES:
                    print(f"Person {tid} now addressing the camera!")
            else:
                consec_count[tid] = 0

            # draw bounding box
            x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))
            color = (0,255,0) if stable else (0,0,255)
            cv2.rectangle(frame, (x1i,y1i), (x2i,y2i), color, 2)

            # draw yaw & pitch text below the box
            info_text = f"Yaw={yaw:.1f}° Pitch={pitch:.1f}°"
            cv2.putText(frame, info_text,
                        (x1i, y2i + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255,255,255), 1)

            # draw LOOK/AWAY label
            cv2.putText(frame,
                        f"ID:{tid} {'LOOK' if stable else 'AWAY'}",
                        (x1i, y1i - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow('Fixed Threshold Look Tracking', frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(r'C:\Users\Julian\vis_impair_projects\data\group_test.mp4')
