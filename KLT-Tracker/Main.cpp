#include "KltTracker.h"

using namespace cv;
using namespace std;

bool drawing_box = false;
Rect box;
bool selected = false;
void create_mouse_callback(int event, int x, int y, int flag, void* param);


int main(int argc, char** argv)
{
	Mat orig_img, temp_img;

	VideoCapture video_capture;
	video_capture = cv::VideoCapture("test.avi");
	video_capture.read(orig_img);

	cv::namedWindow("frame");

	temp_img = orig_img.clone();

	cv::setMouseCallback("frame", create_mouse_callback, (void*)&temp_img);

	cv::imshow("frame", orig_img);

	while (selected == false)
	{
		cv::Mat temp;

		temp_img.copyTo(temp);

		if (drawing_box)
			cv::rectangle(temp, box, cv::Scalar(0), 2);

		cv::imshow("frame", temp);
		if (cv::waitKey(15) == 27)
			break;
	}
	bool affine_refine_flag = true;
	KltTracker klt_tracker(orig_img, box, affine_refine_flag);

	for (;;)
	{
		Mat frame;
		video_capture >> frame;
		if (frame.empty())
			break;
		klt_tracker.track(frame);
		imshow("frame", frame);
		waitKey(1);
	}

	return 0;
}


void create_mouse_callback(int event, int x, int y, int flag, void* param)
{
	cv::Mat *image = (cv::Mat*) param;
	switch (event){
	case CV_EVENT_MOUSEMOVE:
		if (drawing_box){
			box.width = x - box.x;
			box.height = y - box.y;
		}
		break;

	case CV_EVENT_LBUTTONDOWN:
		drawing_box = true;
		box = cv::Rect(x, y, 0, 0);
		break;

	case CV_EVENT_LBUTTONUP:
		drawing_box = false;
		if (box.width < 0){
			box.x += box.width;
			box.width *= -1;
		}
		if (box.height < 0){
			box.y += box.height;
			box.height *= -1;
		}
		cv::rectangle(*image, box, cv::Scalar(0), 2);
		selected = true;
		break;
	}
}


