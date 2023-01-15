#pragma once
#ifndef EVALMETRICS_H
#define EVALMETRICS_H

#include <opencv2/opencv.hpp>


float MSE(const cv::Mat src1, const cv::Mat src2)
{
	cv::Mat src1_copy = src1.clone();
	cv::Mat src2_copy = src2.clone();

	src1_copy.convertTo(src1_copy, CV_32FC1);
	src2_copy.convertTo(src2_copy, CV_32FC1);

	// Compute the mse_numerater
	cv::Mat mse_numerater_matrix;
	cv::absdiff(src1_copy, src2_copy, mse_numerater_matrix);
	mse_numerater_matrix = mse_numerater_matrix.mul(mse_numerater_matrix);
	float mse_numerater = cv::sum(mse_numerater_matrix)[0];

	// Compute the MSE
	float mse_denominator = src1_copy.cols * src1_copy.rows;
	float my_mse = mse_numerater / mse_denominator;

	return my_mse;
}


float AverageValue(const cv::Mat src)
{
	cv::Mat src_copy = src.clone();
	float src_copy_sum = cv::sum(src_copy)[0];
	float src_copy_average = src_copy_sum / (src_copy.cols * src_copy.rows);

	return src_copy_average;
}


float CorrelationCoefficient(const cv::Mat src1, const cv::Mat src2)
{
	cv::Mat src1_copy = src1.clone();
	cv::Mat src2_copy = src2.clone();

	src1_copy.convertTo(src1_copy, CV_32FC1);
	src2_copy.convertTo(src2_copy, CV_32FC1);

	float src_1_copy_average = AverageValue(src1_copy);
	float src_2_copy_average = AverageValue(src2_copy);

	cv::Mat src_1_copy_subtraction = src1_copy - src_1_copy_average;
	cv::Mat src_2_copy_subtraction = src2_copy - src_2_copy_average;

	cv::Mat correlation_coefficient_numerater_matrix = src_1_copy_subtraction.mul(src_2_copy_subtraction);
	float correlation_coefficient_numerater = cv::sum(correlation_coefficient_numerater_matrix)[0];

	cv::Mat correlation_coefficient_denominator_matrix1 = src_1_copy_subtraction.mul(src_1_copy_subtraction);
	cv::Mat correlation_coefficient_denominator_matrix2 = src_2_copy_subtraction.mul(src_2_copy_subtraction);
	float correlation_coefficient_denominator1 = cv::sum(correlation_coefficient_denominator_matrix1)[0];
	float correlation_coefficient_denominator2 = cv::sum(correlation_coefficient_denominator_matrix2)[0];
	float correlation_coefficient_denominator = sqrt(correlation_coefficient_denominator1 * correlation_coefficient_denominator2);

	float my_correlation_coefficient = correlation_coefficient_numerater / correlation_coefficient_denominator;

	return my_correlation_coefficient;
}


float SNR(const cv::Mat src1, const cv::Mat src2)
{
	cv::Mat src1_copy = src1.clone();
	cv::Mat src2_copy = src2.clone();

	src1_copy.convertTo(src1_copy, CV_32FC1);
	src2_copy.convertTo(src2_copy, CV_32FC1);

	// Compute the snr_numerater and snr_denominator
	cv::Mat snr_numerater_matrix;
	snr_numerater_matrix = src1_copy.mul(src1_copy);
	float snr_numerator = cv::sum(snr_numerater_matrix)[0];

	cv::Mat snr_denominator_matrix;
	cv::absdiff(src1_copy, src2_copy, snr_denominator_matrix);
	snr_denominator_matrix = snr_denominator_matrix.mul(snr_denominator_matrix);
	float snr_denominator = cv::sum(snr_denominator_matrix)[0];

	// Compute the SNR
	float my_snr = snr_numerator / snr_denominator;
	my_snr = 10 * log10(my_snr);

	return my_snr;
}


float PSNR(const cv::Mat src1, const cv::Mat src2)
{
	cv::Mat src1_copy = src1.clone();
	cv::Mat src2_copy = src2.clone();

	src1_copy.convertTo(src1_copy, CV_32FC1);
	src2_copy.convertTo(src2_copy, CV_32FC1);

	// Compute the psnr_numerater and psnr_denominator
	float psnr_numerator = 255 * 255;
	float psnr_denominator = MSE(src1_copy, src2_copy);

	// Compute the PSNR
	float my_psnr = psnr_numerator / psnr_denominator;
	my_psnr = 10 * log10(my_psnr);

	return my_psnr;
}


float Entropy(const cv::Mat src)
{
	cv::Mat src_copy = src.clone();
	float gray_array[256] = { 0.0 };

	// Compute the nums of every gray value
	for (int i = 0; i < src_copy.rows; i++)
	{
		uchar* src_copy_poniter = src_copy.ptr<uchar>(i);

		for (int j = 0; j < src_copy.cols; j++)
		{
			int gray_value = src_copy_poniter[j];
			gray_array[gray_value]++;
		}
	}

	// Compute the probability of every gray value and entropy
	float my_entropy = 0;
	float* gray_array_pointer = gray_array;
	for (int i = 0; i < 255; i++)
	{
		float gray_value_prob = *gray_array_pointer / (src_copy.cols * src_copy.rows);

		if (gray_value_prob != 0)
		{
			my_entropy = my_entropy - gray_value_prob * log(gray_value_prob);
		}
		else
		{
			my_entropy = my_entropy;
		}
		gray_array_pointer++;
	}

	return my_entropy;
}


float JointEntropy(const cv::Mat src1, const cv::Mat src2)
{
	cv::Mat src1_copy = src1.clone();
	cv::Mat src2_copy = src2.clone();
	float gray_array[256][256] = { 0.0 };

	// Compute the nums of every gray value pair
	for (int i = 0; i < src1_copy.rows; i++)
	{
		uchar* src1_copy_poniter = src1_copy.ptr<uchar>(i);
		uchar* src2_copy_poniter = src2_copy.ptr<uchar>(i);

		for (int j = 0; j < src1_copy.cols; j++)
		{
			int gray_value1 = src1_copy_poniter[j];
			int gray_value2 = src2_copy_poniter[j];
			gray_array[gray_value1][gray_value2]++;
		}
	}

	// Compute the joint_entropy
	// ()优先级高，说明gray_array_pointer是一个指针，指向一个double类型的一维数组，其长度是256
	// 256也可以说是gray_array_pointer的步长，也就是说执行 p + 1 时，gray_array_poniter要跨过256个double类型的长度
	float my_joint_entropy = 0;
	float(*gray_array_pointer)[256] = gray_array;

	for (int i = 0; i < 255; i++)
	{
		for (int j = 0; j < 255; j++)
		{
			// *(*(gray_array + i) + j)
			float gray_value_prob = gray_array_pointer[i][j] / (src1_copy.cols * src1_copy.rows);

			if (gray_value_prob != 0)
			{
				my_joint_entropy = my_joint_entropy - gray_value_prob * log(gray_value_prob);
			}
			else
			{
				my_joint_entropy = my_joint_entropy;
			}
		}
	}

	return my_joint_entropy;
}

float MutualInformation(const cv::Mat src1, const cv::Mat src2)
{
	cv::Mat src1_copy = src1.clone();
	cv::Mat src2_copy = src2.clone();

	float entropy1 = Entropy(src1_copy);
	float entropy2 = Entropy(src2_copy);
	float joint_entropy = JointEntropy(src1_copy, src2_copy);
	float my_mutual_information = entropy1 + entropy2 - joint_entropy;

	return my_mutual_information;
}


float NormalizedMutualInformation(const cv::Mat src1, const cv::Mat src2)
{
	cv::Mat src1_copy = src1.clone();
	cv::Mat src2_copy = src2.clone();

	float entropy1 = Entropy(src1_copy);
	float entropy2 = Entropy(src2_copy);
	float joint_entropy = JointEntropy(src1_copy, src2_copy);
	float my_normalized_mutual_information = 2 * joint_entropy / (entropy1 + entropy2);

	return my_normalized_mutual_information;
}


float SSIM(const cv::Mat src1, const cv::Mat src2)
{
	cv::Mat src1_copy = src1.clone();
	cv::Mat src2_copy = src2.clone();

	const float c1 = 6.5025, c2 = 58.5225;
	src1_copy.convertTo(src1_copy, CV_32FC1);
	src2_copy.convertTo(src2_copy, CV_32FC1);

	cv::Mat mu1, mu2;
	cv::GaussianBlur(src1_copy, mu1, cv::Size(11, 11), 1.5);	//u_x
	cv::GaussianBlur(src2_copy, mu2, cv::Size(11, 11), 1.5);	//u_y
	cv::Mat mu1_2 = mu1.mul(mu1);	//u_x^2
	cv::Mat mu2_2 = mu2.mul(mu2);	//u_y^2
	cv::Mat mu1_mu2 = mu1.mul(mu2);	//u_x*u_y

	cv::Mat I1_2 = src1_copy.mul(src1_copy);	//x^2
	cv::Mat I2_2 = src2_copy.mul(src2_copy);	//y^2
	cv::Mat I1_I2 = src1_copy.mul(src2_copy);	//x*y

	cv::Mat sigma1_2, sigma2_2, sigma1_sigma2;
	GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
	sigma1_2 -= mu1_2;
	GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
	sigma2_2 -= mu2_2;
	GaussianBlur(I1_I2, sigma1_sigma2, cv::Size(11, 11), 1.5);
	sigma1_sigma2 -= mu1_mu2;

	cv::Mat t1, t2, t3;
	t1 = 2 * mu1_mu2 + c1;
	t2 = 2 * sigma1_sigma2 + c2;
	t3 = t1.mul(t2);
	t1 = mu1_2 + mu2_2 + c1;
	t2 = sigma1_2 + sigma2_2 + c2;
	t1 = t1.mul(t2);
	cv::Mat my_ssim_matrix;
	cv::divide(t3, t1, my_ssim_matrix);

	float my_ssim = cv::mean(my_ssim_matrix)[0];

	return my_ssim;
}


float MSSIM(const cv::Mat src1, const cv::Mat src2)
{
	cv::Mat src1_copy = src1.clone();
	cv::Mat src2_copy = src2.clone();

	int win_size = 11;
	float sigma = 1.5;
	int dynamic_range = 255;
	float K1 = 0.01, K2 = 0.03;

	src1_copy.convertTo(src1_copy, CV_32FC1);
	src2_copy.convertTo(src2_copy, CV_32FC1);

	// 计算两幅图的平均图
	// ux，uy 的每个像素代表以它为中心的滑窗下所有像素的均值(加权) E(X), E(Y)
	cv::Mat mu_x, mu_y;
	cv::GaussianBlur(src1_copy, mu_x, cv::Size(win_size, win_size), sigma);	//u_x
	cv::GaussianBlur(src2_copy, mu_y, cv::Size(win_size, win_size), sigma);	//u_y

	// compute(weighted) variancesand covariances
	// 计算 E(X^2), E(Y^2), E(XY)
	cv::Mat mu_xx, mu_yy, mu_xy;
	cv::GaussianBlur(src1_copy.mul(src1_copy), mu_xx, cv::Size(win_size, win_size), sigma);	//u_xx
	cv::GaussianBlur(src2_copy.mul(src2_copy), mu_yy, cv::Size(win_size, win_size), sigma);	//u_yy
	cv::GaussianBlur(src1_copy.mul(src2_copy), mu_xy, cv::Size(win_size, win_size), sigma);	//u_xy

	// 进行无偏估计
	float conv_norm = win_size * win_size / (win_size * win_size - 1);
	//sigma_xx = E(X^2) - E(X)^2
	cv::Mat sigma_xx = conv_norm * (mu_xx - (mu_x.mul(mu_x)));
	//sigma_yy = E(Y^2) - E(Y)^2
	cv::Mat sigma_yy = conv_norm * (mu_yy - (mu_y.mul(mu_y)));
	//sigma_xy = E(XY) - E(X)E(Y)
	cv::Mat sigma_xy = conv_norm * (mu_xy - (mu_x.mul(mu_y)));

	// paper 中的公式
	float C1 = K1 * K1 * dynamic_range * dynamic_range;
	float C2 = K2 * K2 * dynamic_range * dynamic_range;

	// paper 中的公式
	cv::Mat A1 = 2 * mu_x.mul(mu_y) + C1;
	cv::Mat A2 = 2 * sigma_xy + C2;
	cv::Mat B1 = mu_x.mul(mu_x) + mu_y.mul(mu_y) + C1;
	cv::Mat B2 = sigma_xx + sigma_yy + C2;

	cv::Mat mssim_matrix;
	// 矩阵中对应位置作除法
	cv::divide(A1.mul(A2), B1.mul(B2), mssim_matrix);

	float my_ssim = cv::mean(mssim_matrix)[0];

	return my_ssim;
}

#endif // !EVALMETRICS_H
