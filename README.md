# MeshStereo-Linux
MeshStereo code in C++ for Linux platform

This code has been taken from the author's website http://chizhang.me/ and has been modified slightly to be able to run at my local end. 

How to run, for e.g. -
./MeshStereo data/cones_right.png data/cones_left.png results/disp_cones 60

Please note:-
1) The results are only generated for the right image, so the first image always has to the right image. If the images are swapped in the above command, unexpected result appears. 
2) Please make sure to provide a rough estimate of the max. disparity level (last argument in the above command) according to the image resolution ( for e.g., for cones - 60)


