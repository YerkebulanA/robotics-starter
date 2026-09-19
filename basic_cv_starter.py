"""Basic computer-vision starter script.

Runs live webcam object detection with a pretrained YOLO model.
Press Q to stop.

Install dependencies:
    pip install ultralytics opencv-python
"""

from ultralytics import YOLO
import cv2


def main() -> None:
    model = YOLO("yolo11n.pt")
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam. Check the camera index and permissions.")

    print("Webcam detection started. Press Q to quit.")

    while True:
        success, frame = camera.read()
        if not success:
            break

        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()
        cv2.imshow("Robotics Starter - YOLO Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
