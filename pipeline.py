import cv2
import numpy as np
import mediapipe as mp
import traceback
from collections import deque

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

YAW_THRESH   = 24.0
PITCH_THRESH = 12.0


mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
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
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌  Cannot open video: {video_path}")
        return

    print("✅  Video opened, processing...")
    frame_idx = 0
    
    HISTORY_LEN = 5
    history = deque(maxlen=HISTORY_LEN)
    
    CONSEC_THRESHOLD = 30   # frames
    consec_count = 0

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

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                angles = get_head_pose(lm, (h, w))
                if angles is not None:
                    pitch, yaw, roll = angles
                    
                    # bring pitch into [-180, +180]
                    if pitch > 180:
                        pitch -= 360
                    elif pitch < -180:
                        pitch += 360

                    # now if abs(pitch) > 90, flip it:
                    if pitch > 90:
                        pitch = 180 - pitch
                    elif pitch < -90:
                        pitch = -180 - pitch
                        
                    print(f"[Frame {frame_idx}] raw yaw={yaw:.1f}, raw pitch={pitch:.1f}")
                    
                    # decide if they’re roughly facing the camera
                    is_looking = (abs(yaw)   < YAW_THRESH) \
                            and (abs(pitch) < PITCH_THRESH)

                    history.append(is_looking)
                    
                    stable_looking = sum(history) >= 3
                    
                    cv2.putText(
                        frame,
                        f"Yaw={yaw:.1f}°, Pitch={pitch:.1f}°",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0,255,0),
                        2
                    )
                                        
                    cv2.putText(
                        frame,
                        "LOOKING" if is_looking else "AWAY",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,0,255),
                        2
                    )
                    
                    cv2.putText(
                        frame,
                        "STABLE LOOKING" if stable_looking else "AWAY",
                        (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255,0,0),
                        2
                    )
                    
                    if stable_looking:
                        consec_count += 1
                        if consec_count == CONSEC_THRESHOLD:
                            print("💬 User is now addressing the camera!")
                            # → trigger whatever you need here
                    else:
                        consec_count = 0

                    
            # show your frame
            cv2.imshow("Pose Estimation", frame)

            # press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🛑 Interrupted by user")
                break

        except Exception as e:
            # catch anything that goes wrong and print it
            print(f"⚠️ Error on frame {frame_idx}:")
            traceback.print_exc()
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🏁 Done.")

if __name__ == "__main__":
    main(r"C:\Users\Julian\vis_impair_projects\data\test_group.mp4")
