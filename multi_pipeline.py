import cv2
import numpy as np
import mediapipe as mp
import traceback
import sys
from collections import deque, defaultdict

# add your SORT directory
sys.path.append(r'C:\Users\Julian\vis_impair_projects\sort')
from sort import Sort

# ── CONFIG ────────────────────────────────────────────────────────────────
MAX_FACES             = 3
CALIB_FRAMES_PER_TRACK = 50    # frames to collect neutral head-pose per track
HISTORY_LEN           = 5     # smoothing window for look votes
VOTE_THRESHOLD        = 3     # must have ≥ this many True in HISTORY_LEN
CONSEC_SECONDS        = 1.5   # seconds of continuous "looking" to fire event
EMA_ALPHA             = 0.3   # smoothing on angles
GATING_FACTOR         = 0.5   # fraction of box diagonal for gating
MIN_MAD_THRESH        = 5.0   # minimum degrees for MAD

# 3D model points (nose tip, chin, left eye, right eye, mouth corners)
MODEL_POINTS = np.array([
    (0.0,   0.0,    0.0),     # nose tip
    (0.0, -330.0, -65.0),     # chin
    (-225.0, 170.0, -135.0),  # left eye
    (225.0, 170.0, -135.0),   # right eye
    (-150.0, -150.0, -125.0), # left mouth
    (150.0, -150.0, -125.0)   # right mouth
])
LM_IDX = {"nose":1, "chin":152, "left_eye":33, "right_eye":263, "left_mouth":61, "right_mouth":291}

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=MAX_FACES,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# storage per-track
yaw_hist         = defaultdict(list)
pitch_hist       = defaultdict(list)
calib_count      = defaultdict(int)
track_thresholds = {}            # tid -> (yaw_med, yaw_thr, pitch_med, pitch_thr)
last_seen        = {}            # tid -> last frame index seen


def get_head_pose(landmarks, frame_size):
    """
    Estimate head pose (pitch, yaw, roll) in degrees from landmarks.
    """
    h, w = frame_size
    img_pts = np.array([
        (landmarks[LM_IDX['nose']].x  * w, landmarks[LM_IDX['nose']].y  * h),
        (landmarks[LM_IDX['chin']].x  * w, landmarks[LM_IDX['chin']].y  * h),
        (landmarks[LM_IDX['left_eye']].x * w, landmarks[LM_IDX['left_eye']].y * h),
        (landmarks[LM_IDX['right_eye']].x * w, landmarks[LM_IDX['right_eye']].y * h),
        (landmarks[LM_IDX['left_mouth']].x * w, landmarks[LM_IDX['left_mouth']].y * h),
        (landmarks[LM_IDX['right_mouth']].x * w, landmarks[LM_IDX['right_mouth']].y * h)
    ], dtype='double')

    focal = w
    center = (w/2, h/2)
    cam_mat = np.array([[focal, 0, center[0]],
                        [0, focal, center[1]],
                        [0, 0, 1]], dtype='double')
    dist = np.zeros((4,1))

    success, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS, img_pts, cam_mat, dist,
        flags=cv2.SOLVEPNP_EPNP
    )
    if not success:
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
    x = np.arctan2(rmat[2,1], rmat[2,2])
    y = np.arctan2(-rmat[2,0], sy)
    z = np.arctan2(rmat[1,0], rmat[0,0])
    return np.degrees((x, y, z)), (rvec, tvec, cam_mat, dist)


def compute_thresholds(yaw_vals, pitch_vals):
    y_med = np.median(yaw_vals)
    p_med = np.median(pitch_vals)
    y_mad = np.median(np.abs(yaw_vals - y_med))
    p_mad = np.median(np.abs(pitch_vals - p_med))
    y_thr = max(MIN_MAD_THRESH, 2.5 * y_mad)
    p_thr = max(MIN_MAD_THRESH, 2.5 * p_mad)
    return y_med, y_thr, p_med, p_thr


def main(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('❌ Cannot open video:', video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"▶️ Opened {video_path}: {total_frames} frames @ {fps:.1f} fps")

    delay = int(1000 / fps)
    consec_thresh = int(CONSEC_SECONDS * fps)

    tracker = Sort(max_age=30, min_hits=1, iou_threshold=0.3)
    histories = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
    consec_count = defaultdict(int)
    ema_yaw = {}
    ema_pitch = {}
    drawing_started = False

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("🔚 End of video")
            break
        frame_idx += 1

        try:
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = mp_face_mesh.process(rgb)
            dets, lms = [], []
            if results.multi_face_landmarks:
                for lm_set in results.multi_face_landmarks:
                    xs = [lm.x for lm in lm_set.landmark]
                    ys = [lm.y for lm in lm_set.landmark]
                    x1, x2 = int(min(xs)*w), int(max(xs)*w)
                    y1, y2 = int(min(ys)*h), int(max(ys)*h)
                    dets.append([x1, y1, x2, y2, 1.0])
                    lms.append(lm_set.landmark)

            tracks = tracker.update(np.array(dets)) if dets else []
            print(f"[Frame {frame_idx}] dets={len(dets)} tracks={len(tracks)}")

            # update last_seen
            for *_, tid in tracks:
                tid = int(tid)
                last_seen[tid] = frame_idx

            # purge old tracks
            for tid in list(track_thresholds):
                if frame_idx - last_seen.get(tid, 0) > fps * 5:
                    track_thresholds.pop(tid, None)
                    calib_count.pop(tid, None)
                    yaw_hist.pop(tid, None)
                    pitch_hist.pop(tid, None)

            # per-track calibration
            for x1, y1, x2, y2, tid in tracks:
                tid = int(tid)
                cx, cy = (x1+x2)/2, (y1+y2)/2
                best_i, best_dist = None, float('inf')
                for i,(dx1,dy1,dx2,dy2,_) in enumerate(dets):
                    dcx, dcy = (dx1+dx2)/2, (dy1+dy2)/2
                    d2 = (dcx-cx)**2 + (dcy-cy)**2
                    if d2 < best_dist:
                        best_dist, best_i = d2, i

                if tid not in track_thresholds:
                    if calib_count[tid] < CALIB_FRAMES_PER_TRACK:
                        res = get_head_pose(lms[best_i], (h, w))
                        if res is None: continue
                        (pitch, yaw, _), _ = res
                        if pitch > 180: pitch -= 360
                        elif pitch < -180: pitch += 360
                        pitch = np.clip(pitch, -90, 90)
                        yaw_hist[tid].append(yaw)
                        pitch_hist[tid].append(pitch)
                        calib_count[tid] += 1
                    elif calib_count[tid] == CALIB_FRAMES_PER_TRACK:
                        thr = compute_thresholds(np.array(yaw_hist[tid]), np.array(pitch_hist[tid]))
                        track_thresholds[tid] = thr
                        print(f"Track {tid} calibrated → yaw {thr[0]:.1f}±{thr[1]:.1f}, pitch {thr[2]:.1f}±{thr[3]:.1f}")
                        calib_count[tid] += 1

            if not drawing_started and any(int(tid) in track_thresholds for *_, tid in tracks):
                print(f"▶️ Drawing boxes from frame {frame_idx} onward")
                drawing_started = True

            # drawing & classification
            for x1, y1, x2, y2, tid in tracks:
                tid = int(tid)
                if tid not in track_thresholds:
                    continue

                # cast to ints
                x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))

                # gating
                cx, cy   = (x1+x2)/2, (y1+y2)/2
                box_diag = np.hypot(x2-x1, y2-y1)
                max_d2    = (GATING_FACTOR * box_diag)**2
                best_i, best_d2 = None, float('inf')
                for i,(dx1,dy1,dx2,dy2,_) in enumerate(dets):
                    dcx, dcy = (dx1+dx2)/2, (dy1+dy2)/2
                    d2 = (dcx-cx)**2 + (dcy-cy)**2
                    if d2 < best_d2:
                        best_d2, best_i = d2, i
                if best_d2 > max_d2:
                    histories[tid].append(False)
                else:
                    res = get_head_pose(lms[best_i], (h, w))
                    if res is None:
                        histories[tid].append(False)
                    else:
                        (pitch, yaw, _), _ = res
                        if pitch > 180: pitch -= 360
                        elif pitch < -180: pitch += 360
                        pitch = np.clip(pitch, -90, 90)
                        ema_yaw[tid]   = EMA_ALPHA * yaw   + (1-EMA_ALPHA)*ema_yaw.get(tid, yaw)
                        ema_pitch[tid] = EMA_ALPHA * pitch + (1-EMA_ALPHA)*ema_pitch.get(tid, pitch)
                        y_med, y_thr, p_med, p_thr = track_thresholds[tid]
                        ang_dist = np.hypot(ema_yaw[tid]-y_med, ema_pitch[tid]-p_med)
                        histories[tid].append(ang_dist < np.hypot(y_thr, p_thr))

                stable = sum(histories[tid]) >= VOTE_THRESHOLD
                if stable:
                    consec_count[tid] += 1
                    if consec_count[tid] == consec_thresh:
                        print(f"💬 Track {tid} now addressing the camera!")
                else:
                    consec_count[tid] = 0

                color = (0,255,0) if stable else (0,0,255)
                cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), color, 2)
                cv2.putText(frame,
                            f"ID:{tid} {'LOOK' if stable else 'AWAY'}",
                            (x1i, y1i-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow('Multi Face Look Tracking', frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                print("🛑 Interrupted by user")
                break

        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            traceback.print_exc()
            continue

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(r'C:\Users\Julian\vis_impair_projects\data\group_test.mp4')

# TODO: integrate Gaze360 per-track to refine gaze vector (replace PnP angles)
