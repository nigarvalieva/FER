import cv2
from ultralytics import YOLO
import signal

model = YOLO("best.pt")
running = True

def signal_handler(sig, frame):
    global running
    print("\nStopped by user (keys 'CTRL+C' pressed).")
    running = False

signal.signal(signal.SIGINT, signal_handler)

cap = cv2.VideoCapture(0)

while running:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.25, verbose=False)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO - Press 'CTRL+C' to exit", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopped by user (keys 'CTRL+C' pressed).")
        break

cap.release()
cv2.destroyAllWindows()