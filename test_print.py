import cv2
import numpy as np
import mediapipe as mp

print("▶️  Starting debug_pipeline.py…")

# 3D model points for head pose (nose tip, chin, eyes, mouth corners)
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
])

# Initialize MediaPipe face detection and mesh
print("▶️  Initializing MediaPipe face detection and mesh...")
detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Head-pose estimation function
def get_head_pose(frame):
    img_h, img_w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mesh_results = mesh.process(rgb)
    if not mesh_results.multi_face_landmarks:
        return None
    lm = mesh_results.multi_face_landmarks[0].landmark

    # 2D image points
    image_points = np.array([
        (lm[1].x * img_w, lm[1].y * img_h),     # Nose tip
        (lm[199].x * img_w, lm[199].y * img_h),   # Chin
        (lm[33].x * img_w, lm[33].y * img_h),     # Left eye left corner
        (lm[263].x * img_w, lm[263].y * img_h),   # Right eye right corner
        (lm[61].x * img_w, lm[61].y * img_h),     # Left mouth corner
        (lm[291].x * img_w, lm[291].y * img_h)    # Right mouth corner
    ], dtype="double")

    # Camera internals
    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    success, rot_vec, _ = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    rmat, _ = cv2.Rodrigues(rot_vec)
    sy = np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
    x = np.arctan2(rmat[2, 1], rmat[2, 2])
    y = np.arctan2(-rmat[2, 0], sy)
    z = np.arctan2(rmat[1, 0], rmat[0, 0])
    return np.degrees((x, y, z))  # pitch, yaw, roll

if __name__ == '__main__':
    video_path = r'C:\Users\Julian\vis_impair_projects\data\test_group.mp4'
    print(f"▶️  Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌  Cannot open video {video_path}")
        exit(1)
    print("✅  Video opened, reading first frame...")

    ret, frame = cap.read()
    print("▶️  First frame read:", ret, "shape:", None if not ret else frame.shape)
    if not ret:
        print("❌  Failed to read the first frame—check your file.")
        exit(1)

    # Face detection
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)
    if results.detections:
        print(f"  Detected {len(results.detections)} face(s)")
        d = results.detections[0]
        bb = d.location_data.relative_bounding_box
        h, w, _ = frame.shape
        x1 = int(bb.xmin * w)
        y1 = int(bb.ymin * h)
        x2 = x1 + int(bb.width * w)
        y2 = y1 + int(bb.height * h)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Estimate head pose
        face_crop = frame[y1:y2, x1:x2]
        angles = get_head_pose(face_crop)
        if angles:
            pitch, yaw, roll = angles
            cv2.putText(
                frame,
                f"yaw={yaw:.1f}° pitch={pitch:.1f}°",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
    else:
        print("  No faces detected on first frame")

    # Show single annotated frame
    cv2.imshow('Debug Frame', frame)
    print("▶️  Displaying annotated frame—press any key to continue.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()
    print("🏁  Done single frame debug.")