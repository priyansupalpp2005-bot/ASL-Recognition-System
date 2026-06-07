import cv2

print("Checking cameras...")

for i in range(10):
    cap = cv2.VideoCapture(i)

    if cap.isOpened():
        print(f"Camera found at index {i}")

    cap.release()

print("Done")