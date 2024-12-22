import numpy as np
import cv2 as cv


def test_video_capture():
    # read from the default camera
    cap = cv.VideoCapture(0)
    # read from a video file
    # cap = cv.VideoCapture('vtest.avi')
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Our operations on the frame come here
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv.imshow("frame", gray)
        if cv.waitKey(1) == ord("q"):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


def test_video_write():
    cap = cv.VideoCapture(0)

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*"XVID")
    out = cv.VideoWriter("output.avi", fourcc, 20.0, (640, 480))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = cv.flip(frame, 0)

        # write the flipped frame
        out.write(frame)

        cv.imshow("frame", frame)
        if cv.waitKey(1) == ord("q"):
            break
    # Release everything if job is finished
    cap.release()
    out.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    test_video_capture()
