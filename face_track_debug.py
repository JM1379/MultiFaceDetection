print("▶️ Starting face_track_debug…")

import cv2
print("✔️  cv2 imported, version:", cv2.__version__)

import mediapipe as mp
print("✔️  mediapipe imported")

import numpy as np
print("✔️  numpy imported")

import sys
sys.path.append(r'C:\Users\Julian\vis_impair_projects\sort')

from sort import Sort
print("✔️  SORT imported")

def main(video_path=r'C:\Users\Julian\vis_impair_projects\data\test_group.mp4'):
    print("▶️  Opening video:", video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌  Error opening video—check the path!")
        return

    print("✅  Video opened successfully. Initializing face detector and tracker...")
    mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    tracker = Sort()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("🔚 End of video at frame", frame_idx)
            break

        frame_idx += 1
        print(f"\nFrame {frame_idx}: shape={frame.shape}")

        # Face detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face.process(rgb)
        n_dets = len(results.detections) if results.detections else 0
        print(f"  Detected faces: {n_dets}")

        detections = []
        if results.detections:
            h, w, _ = frame.shape
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = x1 + int(bbox.width * w)
                y2 = y1 + int(bbox.height * h)
                score = det.score[0] if det.score else 1.0
                detections.append([x1, y1, x2, y2, score])
            print("  Detection boxes:", detections)

        # Update tracker
        tracks = tracker.update(np.array(detections) if detections else np.empty((0,5)))
        print(f"  Tracker output ({len(tracks)} tracks):", tracks.astype(int).tolist())

        # Draw
        for x1, y1, x2, y2, track_id in tracks.astype(int):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID:{track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Face Tracking Debug', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑  Interrupted by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
