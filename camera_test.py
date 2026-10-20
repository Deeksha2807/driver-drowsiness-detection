
import cv2
import time

tests = [
    (0, cv2.CAP_DSHOW, "Camera 0 - DirectShow"),
    (1, cv2.CAP_DSHOW, "Camera 1 - DirectShow"),
    (0, cv2.CAP_MSMF, "Camera 0 - MSMF"),
    (1, cv2.CAP_MSMF, "Camera 1 - MSMF"),
]

for index, backend, name in tests:
    print("\nTesting:", name)

    cap = cv2.VideoCapture(index, backend)

    if not cap.isOpened():
        print("Camera could not open")
        cap.release()
        continue

    # Allow camera to initialize
    time.sleep(1)

    for attempt in range(15):
        ret, frame = cap.read()

        if ret and frame is not None:
            mean_brightness = frame.mean()
            print(
                f"Frame received=True, "
                f"brightness={mean_brightness:.2f}, "
                f"size={frame.shape}"
            )

            cv2.imshow(name, frame)
            print("Press Q to close this test window.")

            key = cv2.waitKey(1000) & 0xFF

            if key == ord("q") or key == 27:
                cap.release()
                cv2.destroyAllWindows()
                raise SystemExit

        else:
            print("Frame received=False")

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

print("\nAll camera tests completed.")
