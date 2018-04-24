The files are organized as
joint_positions/[class_name]/[video_name]/joint_positions.mat
Each mat file contains four variables
(1) viewpoint: a string, possible values are E ENE ESE N NE NNE NNW NW S SE SSE SSW SW W WNW WSW
(2) scale: an array whose length equals to the number of frames in the video. the i-th entry is the scale of the person in the i-th frame
(3) pos_img: a 3D matrix with size 2 x 15 x [the number of frames].
In the first dimension, the first and second value correspond to the x and y coordinate, respectively.
In the second dimension, the values are
1: neck
2: belly
3: face
4: right shoulder
5: left  shoulder
6: right hip
7: left  hip
8: right elbow
9: left elbow
10: right knee
11: left knee
12: right wrist
13: left wrist
14: right ankle
15: left ankle

1. See a sample annotation of the 15 positions at http://jhmdb.is.tue.mpg.de/puppet_tool
2. Due to the nature of the puppet annotation tool, all 15 joint positions are available even if they are not annotated when they are occluded or outiside the frame.
In this case, the joints are in the neutral puppet positions.
3. The right and left correspond to the right and left side of the annotated person. For example, a person facing the camera has his right side on the left side of the image, and a person back-facing the camera has his right side on the right side of the image.

(4) pos_world is the normalization of pos_img with respect to the frame size and puppet scale,Â
the formula is as below

pos_world(1,:,:) = (pos_img(1,:,:)/W-0.5)*W/H./scale;
pos_world(2,:,:) = (pos_img(2,:,:)/H-0.5)./scale;

W and H are the width and height of the frame, respectively.
