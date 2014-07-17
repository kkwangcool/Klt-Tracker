#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include <Eigen/Core>
#include <iostream>

using namespace cv;
using namespace std;

class KltTracker
{
public:
	KltTracker(const Mat &image, const Rect &box, bool affine_refine_flag);
	void track(Mat &frame);
	void refine_optical_flow_by_affine_combination(const vector<Point2f> &points_1, vector<Point2f> &points_2, Mat &image);
	Point compute_shift(const vector<Point2f> &points_1, vector<Point2f> &points_2);
private:
	vector<Point2f> prev_points_;
	vector<Point2f> current_points_;
	Rect box_;
	TermCriteria termcrit_;
	Size win_size_;
	Mat prev_gray_image_;
	bool affine_refine_flag_;
	float affine_weight_;
};

struct ComparePoints
{
public:
	bool operator()(const Point2f &point1, const Point2f &point2) const
	{
		return (point1.x < point2.x) || ((point1.x == point2.x) && (point1.y < point2.y));
	}
};
