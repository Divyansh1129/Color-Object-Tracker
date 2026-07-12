import cv2
import numpy as np

camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()

    if not success:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) > 500:
            x, y, w, h = cv2.boundingRect(largest)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cx = x + w // 2
            cy = y + h // 2

            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            cv2.putText(
                frame,
                f"X: {cx} Y: {cy}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

    cv2.imshow("Color Object Tracker", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()