# Dlib

[Dlib C++ Library](http://dlib.net/)

Dlib is a modern C++ toolkit containing machine learning algorithms and tools for creating complex software in C++ to solve real world problems. It is used in both industry and academia in a wide range of domains including robotics, embedded devices, mobile phones, and large high performance computing environments. Dlib's open source licensing allows you to use it in any application, free of charge.

Dlib 是一个现代 C++ 工具包，包含机器学习算法和工具，用于用 C++ 创建复杂的软件来解决现实世界的问题。它在工业界和学术界广泛使用，包括机器人、嵌入式设备、移动电话和大型高性能计算环境。 Dlib的开源许可 允许您在任何应用程序中免费使用它。

```bash
pkg-config --cflags --libs dlib-1

-I/usr/local/include -L/usr/local/lib -ldlib /usr/lib/x86_64-linux-gnu/libsqlite3.so
```

local build and install:

apt install:

`get_frontal_face_detector()`

This function returns an `object_detector` that is configured to find human faces that are looking more or less towards the camera. It is created using the `scan_fhog_pyramid` object.


## python bindings

`class dlib.image_window`

This is a GUI window capable of showing images on the screen.

add_overlay(rectangles, color=rgb_pixel(255,0,0)) -> None

add_overlay(rectangle, color=rgb_pixel(255,0,0)) -> None

add_overlay(full_object_detection, color=rgb_pixel(255,0,0)) -> None

clear_overlay()

get_next_double_click(self: dlib.image_window) -> object

get_next_keypress()

is_closed() -> bool

set_image(img: numpy.ndarray[(rows, cols), int]) -> None

set_title(title: str) -> None

wait_until_closed() -> None

wait_for_keypress(key: str) -> int

Blocks until the user presses the given key or closes the window.

`class dlib.face_recognition_model_v1`

This object maps human faces into 128D vectors where pictures of the same person are mapped near to each other and pictures of different people are mapped far apart. The constructor loads the face recognition model from a file.

```python

defcompute_face_descriptor(

img: numpy.ndarray[(rows, cols, 3), uint8],

face: full_object_detection,

num_jitters: int=0,

padding: float=0.25),

-> dlib.vector

```

Takes an image and a `full_object_detection` that references a face in that image and converts it into a `128D face descriptor`. If `num_jitters>1` then each face will be randomly jittered slightly `num_jitters` times, each run through the 128D projection, and the average used as the face descriptor. Optionally allows to override default padding of 0.25 around the face.

dlib.vector

This object is an array of vector objects.

## shape_predictor_68_face_landmarks

[facial-point-annotations](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)

`shape_predictor`

One Millisecond Face Alignment with an Ensemble of Regression Trees, CVPR 2014

这篇论文解决了单张图像的人脸对齐问题。我们展示了如何使用回归树集成直接从像素强度的稀疏子集估计人脸的关键点位置，从而实现超实时的高质量预测。我们提出了一种基于梯度提升的通用框架，用于学习回归树集成，优化平方误差损失和自然处理缺失或部分标注的数据。我们展示了如何利用适当的先验信息，利用图像数据的结构来帮助高效的特征选择。我们还研究了不同的正则化策略及其在防止过拟合中的重要性。此外，我们分析了训练数据量对预测精度的影响，并探讨了使用合成数据进行数据增强的效果。
