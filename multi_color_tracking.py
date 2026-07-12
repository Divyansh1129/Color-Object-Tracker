import cv2
import numpy as np

camera = cv2.VideoCapture(0)

colors = {
    "Blue": (
        np.array([100, 150, 50]),
        np.array([140, 255, 255]),
        (255, 0, 0),
    ),
    "Green": (
        np.array([40, 70, 70]),
        np.array([80, 255, 255]),
        (0, 255, 0),
    ),
    "Red": (
        np.array([0, 120, 70]),
        np.array([10, 255, 255]),
        (0, 0, 255),
    ),
}

while True:
    success, frame = camera.read()

    if not success:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for name, (lower, upper, color) in colors.items():
        mask = cv2.inRange(hsv, lower, upper)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if contours:
            largest = max(contours, key=cv2.contourArea)

            if cv2.contourArea(largest) > 500:
                x, y, w, h = cv2.boundingRect(largest)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                cx = x + w // 2
                cy = y + h // 2

                cv2.circle(frame, (cx, cy), 5, color, -1)

                cv2.putText(
                    frame,
                    name,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

    cv2.imshow("Multi Color Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()