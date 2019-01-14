# computer-vision-hw2

__Used with Voxel51's 'ETA' environment__ https://www.voxel51.com/

This is the **_second of four_** homework assignments done in EECS 504 - Computer Vision Fundamentals at the University of Michigan

## Topics included:
* Working with images as functions
* Tensors
* Eigendecomposition
* Harris Corner Detector/Keypoint Detection
* Image Stitching/Keypoint Matching
* Steerable Filters

****__Note__: Portions of this assignment were given to students at the beginning, mostly code relevant to the ETA platform. In a sense, this assignment was a "fill-in-the-blanks" task, where students were to use whatever methods they chose the implement the above functionalities (without using pre-existing NumPy/OpenCV methods). __Specifically, the code I wrote can be found in__:
* 'harris.py' - "_get_harris_corner" function, _lines 119-190_ (__Harris Corner Detection/Keypoint Detection & Tensor Construction__)
* 'image_stiching.py'
  - "_get_homography" function, _lines 193-240_ (__Keypoint Matching & Homography__)
  - "_overlap" function, _lines 243-257_ (__Image Stitching__)
* 'steerable_filter.py' - "_apply_steerable_filter" function, _lines 69-97_ (__Intensity/Orientation Matrices__)


Results of the image stitching (for varying number of correspondence points used) located in "out" folder
