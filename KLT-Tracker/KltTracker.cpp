#include "KltTracker.h"

KltTracker::KltTracker(const Mat &image,const Rect &box,bool affine_refine_flag)
{
	TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);
	termcrit_ = termcrit;
	Size subPixWinSize(10, 10),win_size(31, 31);
	win_size_ = win_size;
	const int MAX_COUNT = 500;

	Mat mask = Mat::zeros(image.size(), CV_8UC1);
	mask(box) = 255;

	cvtColor(image, prev_gray_image_, COLOR_BGR2GRAY);
	goodFeaturesToTrack(prev_gray_image_, prev_points_, MAX_COUNT, 0.01, 10, mask, 3, 0, 0.04);
	cornerSubPix(prev_gray_image_, prev_points_, subPixWinSize, Size(-1, -1), termcrit_);

	box_ = box;
	affine_weight_ = 0.01;
	affine_refine_flag_ = affine_refine_flag;
}
Point KltTracker::compute_shift(const vector<Point2f> &points_1, vector<Point2f> &points_2)
{
	const int point_num = points_1.size();
	vector<float> delta_x_vec;
	vector<float> delta_y_vec;
	for (int i = 0; i < point_num; i++)
	{
		delta_x_vec.push_back(points_2[i].x - points_1[i].x);
		delta_y_vec.push_back(points_2[i].y - points_1[i].y);
	}
	sort(delta_x_vec.begin(), delta_x_vec.end());
	sort(delta_y_vec.begin(), delta_y_vec.end());
	int delta_x = round(delta_x_vec[point_num / 2]);
	int delta_y = round(delta_y_vec[point_num / 2]);
	return Point(delta_x, delta_y);
}
void KltTracker::refine_optical_flow_by_affine_combination(const vector<Point2f> &points_1, vector<Point2f> &points_2, Mat &image)
{
	map<Point2f, int, ComparePoints> point_index_map;
	const int point_num = points_1.size();
	for (int i = 0; i < point_num; i++)
	{
		point_index_map[points_1[i]] = i;
	}

	Rect rect(0, 0, image.size().width, image.size().height);
	Subdiv2D subdiv(rect);
	subdiv.insert(points_1);
	vector<Vec4f> edgeList;
	subdiv.getEdgeList(edgeList);

	Mat neighbor_mat = Mat::zeros(point_num, point_num, CV_8UC1);
	for (size_t i = 0; i < edgeList.size(); i++)
	{
		Vec4f e = edgeList[i];

		Point2f pt1 = Point2f(e[0], e[1]);
		Point2f pt2 = Point2f(e[2], e[3]);
		if (!(rect.contains(pt1) && rect.contains(pt2)))
			continue;
		int pt1_index = point_index_map[pt1];
		int pt2_index = point_index_map[pt2];
		neighbor_mat.at<char>(pt1_index, pt2_index) = 1;
		neighbor_mat.at<char>(pt2_index, pt1_index) = 1;
		line(image, pt1, pt2, Scalar(255, 255, 255), 1, CV_AA, 0);
	}

	Mat coefficient_mat = Mat::zeros(point_num, point_num, CV_32FC1);
	for (int i = 0; i < point_num; i++)
	{


		int neighbor_num = sum(neighbor_mat.row(i))[0];
		Eigen::MatrixXd A;
		A.resize(3, neighbor_num);
		char* neighbor_mat_pointer = neighbor_mat.ptr<char>(i);
		int k = 0;
		for (int j = 0; j < point_num; j++)
		{
			if (neighbor_mat_pointer[j])
			{
				A(0, k) = points_1[j].x;
				A(1, k) = points_1[j].y;
				A(2, k) = 1.0;
				k++;
			}
		}
		Eigen::MatrixXd B;
		B.resize(3, 1);
		B(0, 0) = points_1[i].x;
		B(1, 0) = points_1[i].y;
		B(2, 0) = 1.0;

		Eigen::MatrixXd W;
		W = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(B);

		k = 0;
		for (int j = 0; j < point_num; j++)
		{
			if (neighbor_mat_pointer[j])
			{
				coefficient_mat.at<float>(j, i) = W(k, 0);
				k++;
			}
		}
	}

	Mat I = Mat::eye(coefficient_mat.size(), CV_32FC1);
	Mat Y(2, point_num, CV_32FC1);

	float* Y_pointer_x = Y.ptr<float>(0);
	float* Y_pointer_y = Y.ptr<float>(1);
	for (int i = 0; i < point_num; i++)
	{
		Y_pointer_x[i] = points_2[i].x;
		Y_pointer_y[i] = points_2[i].y;
	}
	Mat tmp = affine_weight_*(I - coefficient_mat)*(I - coefficient_mat.t()) + I;
	Mat refined_Y = Y*tmp.inv();
	float* refined_Y_pointer_x = refined_Y.ptr<float>(0);
	float* refined_Y_pointer_y = refined_Y.ptr<float>(1);
	for (int i = 0; i < point_num; i++)
	{
		points_2[i].x = refined_Y_pointer_x[i];
		points_2[i].y = refined_Y_pointer_y[i];
	}

	
}

void KltTracker::track(Mat &frame)
{
	Mat current_gray;
	cvtColor(frame, current_gray, COLOR_BGR2GRAY);
	vector<uchar> status;
	vector<float> err;
	calcOpticalFlowPyrLK(prev_gray_image_, current_gray, prev_points_, current_points_, status, err, win_size_,
		3, termcrit_, 0, 0.001);
	int i, k;
	for (i = k = 0; i < current_points_.size(); i++)
	{
		if (!status[i])
			continue;
		prev_points_[k] = prev_points_[i];
		current_points_[k++] = current_points_[i];
	}
	prev_points_.resize(k);
	current_points_.resize(k);
	if (affine_refine_flag_)
	{
		refine_optical_flow_by_affine_combination(prev_points_, current_points_, frame);
		
	}
	Point shift = compute_shift(prev_points_, current_points_);
	box_ += shift;
	for (i = k = 0; i < current_points_.size(); i++)
	{
		if (!box_.contains(current_points_[i]))
			continue;
		current_points_[k++] = current_points_[i];
	}
	current_points_.resize(k);
	std::swap(current_points_, prev_points_);
	prev_gray_image_ = current_gray;
	cv::rectangle(frame, box_, cv::Scalar(0), 2);
	for (int i = 0; i < current_points_.size(); i++)
	{
		circle(frame, current_points_[i], 2, Scalar(255, 0, 0), 2, CV_AA);
	}
}