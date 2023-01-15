#pragma once
#ifndef PADDED_H
#define PADDED_H

#include <opencv2/opencv.hpp>


bool ImagePadded(const cv::Mat& input1, const cv::Mat& input2, cv::Mat& output1, cv::Mat& output2)
{
	int rows_max, cols_max;
	input1.rows > input2.rows ? rows_max = input1.rows : rows_max = input2.rows;
	input1.cols > input2.cols ? cols_max = input1.cols : cols_max = input2.cols;

	cv::copyMakeBorder(input1, output1, 0, rows_max - input1.rows, 0,
		cols_max - input1.cols, cv::BORDER_REPLICATE);

	cv::copyMakeBorder(input2, output2, 0, rows_max - input2.rows, 0,
		cols_max - input2.cols, cv::BORDER_REPLICATE);

	return 0;
}

#endif // !PADDED_H
