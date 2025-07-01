import cv2
import mediapipe as mp
import numpy as np
import sys
sys.path.append(r'C:\Users\Julian\vis_impair_projects\sort')
from sort import Sort

def main(video_path='data/test_group.mp4'):
    mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    tracker = Sort()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("▶️  End of video reached or frame not read.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face.process(rgb)
        detections = []

        h, w, _ = frame.shape
        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = x1 + int(bbox.width * w)
                y2 = y1 + int(bbox.height * h)
                score = det.score[0] if det.score else 1.0
                detections.append([x1, y1, x2, y2, score])

        tracks = tracker.update(np.array(detections))

        for x1, y1, x2, y2, track_id in tracks.astype(int):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID:{track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Face Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
