import cv2

cap = cv2.VideoCapture('data/test_group.mp4')  # adjust path if needed
if not cap.isOpened():
    print("❌  Error opening video—check your path!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("▶️  End of video reached or frame not read.")
        break
    cv2.imshow('Raw Video Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
