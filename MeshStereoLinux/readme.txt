This project provides an implementaion of our stereo model developed under Microsoft Visual Studio 2013. To compile the project, you need to configure OpenCV. We use OpenCV 2.4.8 in the implementation, however others versions greater than 2.4.0 should also work fine.

Usage:
	MeshStereo.exe filePathImageL filePathImageR filePathDispOut numDisps

To see a demo, run:
	MeshStereo.exe im0.png im1.png disp.png 128
You should get two output files:
	disp.png_colorjet_dispL.png
	disp.png_SLIC.png
which are also included in the current folder.

