# opencv-python

haarcascade_frontalface_default.xml: Trained XML classifiers describes some features of some object we want to detect a cascade function is trained from a lot of positive(faces) and negative(non-faces) images.

### Image

Read an image from file (using `cv::imread`)
Display an image in an OpenCV window (using `cv::imshow`)
Write an image to a file (using `cv::imwrite`)

`IMREAD_COLOR` loads the image in the BGR 8-bit format. This is the default that is used here.
`IMREAD_UNCHANGED` loads the image as is (including the alpha channel if present)
`IMREAD_GRAYSCALE` loads the image as an intensity one

`core` section, as here are defined the basic building blocks of the library
`imgcodecs` module, which provides functions for reading and writing
`highgui` module, as this contains the functions to show an image in a window

### Video

VideoCapture

VideoWriter

### Drawing

- Learn to draw different geometric shapes with OpenCV
- You will learn these functions :
  - cv.line(), cv.circle() , cv.rectangle(), cv.ellipse(), cv.putText() etc.

img, color, thikness, linType

### Mouse

Learn to handle mouse events in OpenCV
You will learn these functions : `cv.setMouseCallback()`

['EVENT_FLAG_ALTKEY', 'EVENT_FLAG_CTRLKEY', 'EVENT_FLAG_LBUTTON', 'EVENT_FLAG_MBUTTON', 'EVENT_FLAG_RBUTTON', 'EVENT_FLAG_SHIFTKEY', 'EVENT_LBUTTONDBLCLK', 'EVENT_LBUTTONDOWN', 'EVENT_LBUTTONUP', 'EVENT_MBUTTONDBLCLK', 'EVENT_MBUTTONDOWN', 'EVENT_MBUTTONUP', 'EVENT_MOUSEHWHEEL', 'EVENT_MOUSEMOVE', 'EVENT_MOUSEWHEEL', 'EVENT_RBUTTONDBLCLK', 'EVENT_RBUTTONDOWN', 'EVENT_RBUTTONUP']

### Trackbar

For `cv.createTrackbar()` function, first argument is the trackbar name, second one is the window name to which it is attached, third argument is the default value, fourth one is the maximum value and fifth one is the callback function which is executed every time trackbar value changes. The callback function always has a default argument which is the trackbar position. In our case, function does nothing, so we simply pass.

Another important application of trackbar is to use it as a button or switch. OpenCV, by default, doesn't have button functionality. So you can use trackbar to get such functionality. In our application, we have created one switch in which application works only if switch is ON, otherwise screen is always black.

## Operations on Images

- Basic Operations on Images

  - Learn to read and edit pixel values, working with image ROI and other basic operations.

- Arithmetic Operations on Images

  - Perform arithmetic operations on images

- Performance Measurement and Improvement Techniques

  - Getting a solution is important. But getting it in the fastest way is more important. Learn to check the speed of your code, optimize the code etc.

### Image Processing

### Feature Detection and Description

### Video Analysis
