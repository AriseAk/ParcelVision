import cv2 as cv
import time

def main():
    cap=cv.VideoCapture(0)
    print(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    print(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    prev_time = 0
    while True:
        ret,frame=cap.read()
        if not ret:
            print("Failed")
            break

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        cv.putText(frame, f"FPS: {int(fps)}", (500, 30),cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv.imshow("frame",frame)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()