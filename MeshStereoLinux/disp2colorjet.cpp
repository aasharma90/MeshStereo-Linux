#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cstdio>
#include <algorithm>


// read vmin, vmax from calib file
void readVizRange(char *calibfile, float& dmin, float& dmax)
{
	char line[1000];
	float f;

	dmin = INFINITY;
	dmax = INFINITY;
	FILE *fp = fopen(calibfile, "r");
	if (fp != NULL) {
		while (fgets(line, sizeof line, fp) != NULL) {
			if (sscanf(line, " vmin= %f", &f) == 1) dmin = f;
			if (sscanf(line, " vmax= %f", &f) == 1) dmax = f;
		}
		fclose(fp);
	}
	else
		fprintf(stderr, "Cannot open calib file %s\n", calibfile);
	if (dmin == INFINITY || dmax == INFINITY)
		fprintf(stderr, "Cannot extract vmin, vmax from calib file %s\n", calibfile);
	//printf("read vmin=%f, vmax=%f\n", dmin, dmax);
}

// translate value x in [0..1] into color triplet using "jet" color map
// if out of range, use darker colors
// variation of an idea by http://www.metastine.com/?p=7
void jet(float x, int& r, int& g, int& b)
{
	if (x < 0) x = -0.05;
	if (x > 1) x = 1.05;
	x = x / 1.15 + 0.1; // use slightly asymmetric range to avoid darkest shades of blue.
	r = std::max(0, std::min(255, (int)(round(255 * (1.5 - 4 * fabs(x - .75))))));
	g = std::max(0, std::min(255, (int)(round(255 * (1.5 - 4 * fabs(x - .5))))));
	b = std::max(0, std::min(255, (int)(round(255 * (1.5 - 4 * fabs(x - .25))))));
}

void gray(float x, int& gray_val)
{
	if (x < 0) x = -0.05;
	if (x > 1) x = 1.05;
	x = x / 1.15 + 0.1; 
	gray_val = std::max(0, std::min(255, (int)(round(255 * x))));
}


/*
// get min and max (non-INF) values
void getMinMax(CFloatImage fimg, float& vmin, float& vmax)
{
	CShape sh = fimg.Shape();
	int width = sh.width, height = sh.height;

	vmin = INFINITY;
	vmax = -INFINITY;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			float f = fimg.Pixel(x, y, 0);
			if (f == INFINITY)
				continue;
			vmin = std::min(f, vmin);
			vmax = std::max(f, vmax);
		}
	}
}
*/

// convert float disparity image into a color image using jet colormap
cv::Mat Float2ColorJet(cv::Mat &fimg, float dmin, float dmax)
{
	
	int width = fimg.cols, height = fimg.rows;
	cv::Mat img(height, width, CV_8UC3);

	float scale = 1.0 / (dmax - dmin);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			float f = fimg.at<float>(y, x);
			int r = 0;
			int g = 0;
			int b = 0;

			if (f != INFINITY) {
				float val = scale * (f - dmin);
				jet(val, r, g, b);
			}

			img.at<cv::Vec3b>(y, x) = cv::Vec3b(b, g, r);
		}
	}

	return img;
}

// convert float disparity image into grayscale
cv::Mat Float2Gray(cv::Mat &fimg, float dmin, float dmax)
{
	
	int width = fimg.cols, height = fimg.rows;
	cv::Mat img(height, width, CV_8UC1);

	float scale = 1.0 / (dmax - dmin);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			float f = fimg.at<float>(y, x);
			int gray_val = 0;

			if (f != INFINITY) {
				float val = scale * (f - dmin);
				gray(val, gray_val);
			}

			img.at<uchar>(y, x) = gray_val;
		}
	}

	return img;
}
