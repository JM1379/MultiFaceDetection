import cv2
import numpy as np
import mediapipe as mp
import traceback
from collections import deque, defaultdict
import sys
sys.path.append(r'C:\Users\Julian\vis_impair_projects\sort')
from sort import Sort
print("✔︎ SORT imported successfully")

# 3D model points.
MODEL_POINTS = np.array([
    (0.0,   0.0,    0.0),   # Nose tip
    (0.0, -330.0, -65.0),   # Chin
    (-225.0, 170.0, -135.0),# Left eye outer corner
    (225.0, 170.0, -135.0), # Right eye outer corner
    (-150.0, -150.0, -125.0),# Left mouth corner
    (150.0, -150.0, -125.0) # Right mouth corner
])

LM_IDX = {
    "nose": 1,
    "chin": 152,
    "left_eye": 33,
    "right_eye": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}

YAW_THRESH   = 30.0
PITCH_THRESH = 15.0


mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=3,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_head_pose(landmarks, frame_size):
    h, w = frame_size
    # build 2D image points
    image_points = np.array([
        (landmarks[LM_IDX["nose"]].x  * w, landmarks[LM_IDX["nose"]].y  * h),
        (landmarks[LM_IDX["chin"]].x  * w, landmarks[LM_IDX["chin"]].y  * h),
        (landmarks[LM_IDX["left_eye"]].x  * w, landmarks[LM_IDX["left_eye"]].y  * h),
        (landmarks[LM_IDX["right_eye"]].x * w, landmarks[LM_IDX["right_eye"]].y * h),
        (landmarks[LM_IDX["left_mouth"]].x  * w, landmarks[LM_IDX["left_mouth"]].y  * h),
        (landmarks[LM_IDX["right_mouth"]].x * w, landmarks[LM_IDX["right_mouth"]].y * h)
    ], dtype="double")

    # camera internals from full frame
    focal_length = w
    center = (w/2, h/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4,1))  # assume no lens distortion

    success, rot_vec, _ = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    # convert rotation vector to Euler angles
    rmat, _ = cv2.Rodrigues(rot_vec)
    sy = np.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
    x = np.arctan2(rmat[2,1], rmat[2,2])
    y = np.arctan2(-rmat[2,0], sy)
    z = np.arctan2(rmat[1,0], rmat[0,0])
    return np.degrees((x, y, z))  # pitch, yaw, roll

def main(video_path):
    
    print(f"📹 Processing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌  Cannot open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay = max(1, int(1000 / fps))

    # ── Initialize SORT tracker ──
    tracker = Sort(max_age=15, min_hits=1, iou_threshold=0.3)

    # ── Per-track histories ──
    HISTORY_LEN     = 5
    VOTE_THRESHOLD  = 3    # out of HISTORY_LEN
    CONSEC_THRESHOLD= 30   # frames for a sustained look
    histories  = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
    consec_cnt = defaultdict(int)

    print("✅  Video opened, processing...")
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

            # 1) build detection list (for SORT) and keep landmarks in parallel
            dets = []
            landmarks_list = []
            if results.multi_face_landmarks:
                for lm_set in results.multi_face_landmarks:
                    xs = [lm.x for lm in lm_set.landmark]
                    ys = [lm.y for lm in lm_set.landmark]
                    x1, x2 = min(xs)*w, max(xs)*w
                    y1, y2 = min(ys)*h, max(ys)*h
                    dets.append([x1, y1, x2, y2, 1.0])  # score=1.0
                    landmarks_list.append(lm_set.landmark)

            # 2) update SORT and get back tracks [[x1,y1,x2,y2,track_id], ...]
            if dets:
                tracks = tracker.update(np.array(dets))
            else:
                tracks = []

            # 3) for each track, find matching landmarks by minimal center-distance
            for x1, y1, x2, y2, track_id in tracks:
                cx, cy = (x1+x2)/2, (y1+y2)/2

                # find the detection whose center is closest to this track center
                best_i, best_dist = None, float("inf")
                for i, d in enumerate(dets):
                    dx = (d[0]+d[2])/2 - cx
                    dy = (d[1]+d[3])/2 - cy
                    dist = dx*dx + dy*dy
                    if dist < best_dist:
                        best_dist, best_i = dist, i

                lms = landmarks_list[best_i]
                angles = get_head_pose(lms, (h, w))
                if angles is None:
                    continue
                pitch, yaw, roll = angles

                # normalize pitch into [-90, +90]
                if pitch > 180:      pitch -= 360
                elif pitch < -180:   pitch += 360
                if pitch > 90:       pitch = 180 - pitch
                elif pitch < -90:    pitch = -180 - pitch

                # decide “is looking”
                is_looking = (abs(yaw)   < YAW_THRESH) \
                          and (abs(pitch) < PITCH_THRESH)

                # update history & stable decision
                hist = histories[track_id]
                hist.append(is_looking)
                stable = sum(hist) >= VOTE_THRESHOLD

                # update consecutive-look counter
                if stable:
                    consec_cnt[track_id] += 1
                else:
                    consec_cnt[track_id] = 0

                # trigger when sustained
                if consec_cnt[track_id] == CONSEC_THRESHOLD:
                    print(f"💬 Track {int(track_id)} is now addressing the camera!")

                # draw box, ID and statuses
                col = (0,255,0) if stable else (0,0,255)
                cv2.rectangle(frame,
                              (int(x1),int(y1)),
                              (int(x2),int(y2)), col, 2)
                cv2.putText(frame,
                            f"ID:{int(track_id)} {'LOOK' if stable else 'AWAY'}",
                            (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

            # show frame
            #cv2.imshow("Group Pose Estimation", frame)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    print("🛑 Interrupted by user")
            #    break
            
            cv2.imshow("Group Pose Estimation", frame)
            # wait according to video fps
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                print("🛑 Interrupted by user")
                break
            
        except Exception:
            print(f"⚠️ Error on frame {frame_idx}:")
            traceback.print_exc()
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🏁 Done.")

if __name__ == "__main__":
    main(r"C:\Users\Julian\vis_impair_projects\data\group_test.mp4")
