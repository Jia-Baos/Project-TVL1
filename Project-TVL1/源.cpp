#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/video.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "Myutils.h"
#include "Padded.h"
#include "FlowShow.h"
#include "Akaze.h"
#include "EvalMetrics.h"

int main()
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
	std::cout << "Version: " << CV_VERSION << std::endl;

	//std::string fixed_image_path = "D:\\Code-VS\\picture\\data\\fixed_image1.jpg";
	//std::string moved_image_path = "D:\\Code-VS\\picture\\data\\moved_image1.jpg";
	//std::string fixed_image_path = "D:\\Code-VS\\picture\\data\\dataset7-test1.jpg";
	//std::string moved_image_path = "D:\\Code-VS\\picture\\data\\dataset7-test2.jpg";
	//std::string fixed_image_path = "D:\\Code-VS\\picture\\data\\coke3.jpg";
	//std::string moved_image_path = "D:\\Code-VS\\picture\\data\\coke2.jpg";
	//std::string fixed_image_path = "D:\\DataSet\\dataset3\\template\\template1.png";
	//std::string moved_image_path = "D:\\DataSet\\dataset3\\template\\template2.png";
	std::string fixed_image_path = "E:\\Paper\\OpticalFlowData\\other-data\\Dimetrodon\\frame10.png";
	std::string moved_image_path = "E:\\Paper\\OpticalFlowData\\other-data\\Dimetrodon\\frame11.png";

	cv::Mat moved_image = cv::imread(moved_image_path);
	cv::Mat fixed_image = cv::imread(fixed_image_path);

	cv::cvtColor(moved_image, moved_image, cv::COLOR_BGR2GRAY);
	cv::cvtColor(fixed_image, fixed_image, cv::COLOR_BGR2GRAY);

	ImagePadded(moved_image, fixed_image, moved_image, fixed_image);

	// 进行射影变换
	//std::vector<cv::Point2f> image1_inliers, image2_inliers;
	//AKazeRegistration(moved_image, fixed_image, image1_inliers, image2_inliers);
	//cv::Mat homography = cv::findHomography(image1_inliers, image2_inliers, cv::RANSAC);
	//cv::warpPerspective(moved_image, moved_image, homography, fixed_image.size());

	cv::namedWindow("fixed_image", cv::WINDOW_NORMAL);
	cv::imshow("fixed_image", fixed_image);
	cv::namedWindow("moved_image", cv::WINDOW_NORMAL);
	cv::imshow("moved_image", moved_image);
	cv::namedWindow("Res1", cv::WINDOW_NORMAL);
	cv::imshow("Res1", abs(moved_image - fixed_image));

	cv::Mat flow(fixed_image.size(), CV_32FC2);
	cv::Ptr<cv::optflow::DualTVL1OpticalFlow> tvl1 = cv::optflow::DualTVL1OpticalFlow::create();
	tvl1->calc(fixed_image, moved_image, flow);
	std::cout << "flow.cols: " << flow.cols << "; flow.rows: " << flow.rows << "; flow.channels: " << flow.channels() << std::endl;

	std::vector<cv::Mat> flow_spilit;
	cv::split(flow, flow_spilit);

	// 通过后面一帧重建前面一帧
	cv::Mat result;
	movepixels_2d2(moved_image, result, flow_spilit[0], flow_spilit[1], cv::INTER_CUBIC);

	//光流可视化
	cv::Mat flowrgb(flow.size(), CV_8UC3);
	flo2img(flow, flowrgb);

	cv::namedWindow("result", cv::WINDOW_NORMAL);
	cv::imshow("result", result);
	cv::namedWindow("Res2", cv::WINDOW_NORMAL);
	cv::imshow("Res2", abs(result - fixed_image));
	cv::namedWindow("flowrgb", cv::WINDOW_NORMAL);
	cv::imshow("flowrgb", abs(flowrgb));

	// 评价指标
	std::cout << "********************TVL1 Eval********************" << std::endl;
	std::cout << "MSE:" << MSE(fixed_image, result) << std::endl;
	std::cout << "CC:" << CorrelationCoefficient(fixed_image, result) << std::endl;
	std::cout << "PSNR:" << PSNR(fixed_image, result) << std::endl;
	std::cout << "NMI:" << NormalizedMutualInformation(fixed_image, result) << std::endl;
	std::cout << "MSSIM:" << MSSIM(fixed_image, result) << std::endl;
	std::cout << "********************Eval********************" << std::endl;

	//cv::imwrite("tvl1-res.png", abs(result - fixed_image));
	//cv::imwrite("tvl1-optical-flow.png", flowrgb);

	cv::waitKey();
	return 0;
}