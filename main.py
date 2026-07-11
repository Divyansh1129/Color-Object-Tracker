import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Press 'q' to exit.")

while True:
    success, frame = camera.read()

    if not success:
        print("Error: Failed to capture frame.")
        break

    cv2.imshow("Color Object Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()