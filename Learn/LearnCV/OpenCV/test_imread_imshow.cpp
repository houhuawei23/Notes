/*
g++ test.cpp -o test `pkg-config --cflags --libs opencv4`

pkg-config --cflags opencv4: Returns the compiler flags needed for OpenCV.
pkg-config --libs opencv4: Returns the linker flags needed for OpenCV.
-I/usr/include/opencv4: Specifies the directory for header files.
-L/usr/lib: Specifies the directory for libraries.
-lopencv_core -lopencv_imgcodecs: Links against specific OpenCV modules.


pkg-config

Provide the details of installed libraries for compiling applications.
More information: https://www.freedesktop.org/wiki/Software/pkg-config/.

 - Get the list of libraries and their dependencies:
   pkg-config --libs library1 library2 ...

 - Get the list of libraries, their dependencies, and proper cflags for gcc:
   pkg-config --cflags --libs library1 library2 ...

 - Compile your code with libgtk-3, libwebkit2gtk-4.0 and all their dependencies:
   c++ example.cpp $(pkg-config --cflags --libs gtk+-3.0 webkit2gtk-4.0) -o example
 */

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>

using namespace cv;

int main(int argc, char* argv[]) {
  // get filename from command line
  std::string file_name = argv[1];

  std::string image_path = samples::findFile(file_name);
  Mat img = imread(image_path, IMREAD_COLOR);
  if (img.empty()) {
    std::cout << "Could not read the image: " << image_path << std::endl;
    return 1;
  }
  imshow("Display window", img);
  int k = waitKey(0);  // Wait for a keystroke in the window
  if (k == 's') {
    imwrite("starry_night.png", img);
  }
  return 0;
}
