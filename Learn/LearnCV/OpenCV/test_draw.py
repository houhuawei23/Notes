import numpy as np
import cv2 as cv


drawing = False  # true if mouse is pressed
mode = True  # if True, draw rectangle. Press 'm' to toggle to curve
ix, iy = -1, -1


# mouse callback function
def callback_draw(event, x, y, flags, img):
    global ix, iy, drawing, mode

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
            else:
                cv.circle(img, (x, y), 5, (0, 0, 255), -1)

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
        else:
            cv.circle(img, (x, y), 5, (0, 0, 255), -1)


def draw_some(img):
    # Draw a diagonal blue line with thickness of 5 px
    cv.line(img, (0, 0), (511, 511), (255, 0, 0), 5)

    cv.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)

    # circle
    cv.circle(img, (447, 63), 63, (0, 0, 255), -1)

    # ellipse
    cv.ellipse(img, (256, 256), (100, 50), 0, 0, 360, (255, 0, 255), -1)

    # polygon
    pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv.polylines(img, [pts], True, (0, 255, 255))

    # text
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(img, "OpenCV", (10, 500), font, 4, (255, 255, 255), 2, cv.LINE_AA)


if __name__ == "__main__":

    img = np.zeros((512, 512, 3), np.uint8)
    draw_some(img)
    cv.namedWindow("image")
    cv.setMouseCallback("image", callback_draw, img)

    while 1:
        cv.imshow("image", img)
        k = cv.waitKey(1) & 0xFF
        if k == ord("m"):
            mode = not mode
        elif k == 27:  # ESC
            break

    cv.destroyAllWindows()
