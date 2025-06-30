import cv2
import numpy as np
import mediapipe as mp

print("▶️  Starting head_pose.py…")

# 3D model points of facial landmarks in a generic “head” model.
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),           # Nose tip
    (0.0, -330.0, -65.0),      # Chin
    (-225.0, 170.0, -135.0),   # Left eye left corner
    (225.0, 170.0, -135.0),    # Right eye right corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0, -150.0, -125.0)    # Right mouth corner
])

# Initialize MediaPipe Face Mesh once
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False)

def get_head_pose(frame):
    img_h, img_w = frame.shape[:2]
    results = mp_face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None

    lm = results.multi_face_landmarks[0].landmark

    # 2D image points
    image_points = np.array([
        (lm[1].x * img_w, lm[1].y * img_h),    # Nose tip
        (lm[199].x * img_w, lm[199].y * img_h), # Chin
        (lm[33].x * img_w, lm[33].y * img_h),   # Left eye left corner
        (lm[263].x * img_w, lm[263].y * img_h), # Right eye right corner
        (lm[61].x * img_w, lm[61].y * img_h),   # Left mouth corner
        (lm[291].x * img_w, lm[291].y * img_h)  # Right mouth corner
    ], dtype="double")

    # Camera internals
    focal_length = img_w
    center = (img_w/2, img_h/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4,1))

    success, rotation_vector, _ = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    # Convert rotation vector to Euler angles
    rmat, _ = cv2.Rodrigues(rotation_vector)
    sy = np.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
    x = np.arctan2(rmat[2,1], rmat[2,2])
    y = np.arctan2(-rmat[2,0], sy)
    z = np.arctan2(rmat[1,0], rmat[0,0])
    # x = pitch, y = yaw, z = roll
    return np.degrees((x, y, z))

# ───────────────────────────────────────────────────────
if __name__ == "__main__":
    video_path = r'C:\Users\Julian\vis_impair_projects\data\test_group.mp4'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌  Couldn’t open video:", video_path)
        exit()
    print("✅  Video opened, entering loop")
    
    while True:
        ret, frame = cap.read()
        print("▶️ cap.read() returned:", ret)
        if not ret:
            print("🔚 End of video")
            break

        # Compute head‐pose
        angles = get_head_pose(frame)
        if angles:
            pitch, yaw, roll = angles
            cv2.putText(frame,
                        f"yaw={yaw:.1f}°  pitch={pitch:.1f}°",
                        (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,255,0), 2)

        # Show the annotated frame
        cv2.imshow("Head Pose", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Interrupted by user.")
            break
            
            

    cap.release()
    cv2.destroyAllWindows()
    print("🏁  Done.")
