import cv2

cap = cv2.VideoCapture(0)

print("Camera opened:", cap.isOpened())

ret, frame = cap.read()

print("Frame captured:", ret)

if ret:
    cv2.imshow("Test Camera", frame)
    cv2.waitKey(5000)

cap.release()
cv2.destroyAllWindows()