#pragma once
#ifndef AKAZE_H
#define AKAZE_H

#include <opencv2/opencv.hpp>


//阈值处理+RANSAC
bool AKazeRegistration(const cv::Mat image_object, const cv::Mat image_scene,
	std::vector<cv::Point2f>& object_keypoints_ransac_inliers, std::vector<cv::Point2f>& scene_keypoints_ransac_inliers)
{
	//-- Step 1: Detect the keypoints using AKaze Detector, compute the descriptors
	//提取特征点方法
	cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();

	// 特征点
	// angle：角度，表示关键点的方向
	// class_id：当要对图片进行分类时，我们可以用class_id对每个特征点进行区分，未设定时为-1，需要靠自己设定
	// octave：表示关键点来自金字塔的哪一层
	// pt：关键点的坐标
	// response：关键点的响应程度
	// size：关键点直径的大小
	std::vector<cv::KeyPoint> keypoints_object, keypoints_scene;

	// 单独提取特征点
	// 并将特征点的坐标分别存入keypoints_object, keypoints_scene中
	detector->detect(image_object, keypoints_object);
	detector->detect(image_scene, keypoints_scene);

	// 创建Mat容器用来绘制关键点
	cv::Mat image_object_keypoints;
	cv::Mat image_scene_keypoints;
	cv::drawKeypoints(image_object, keypoints_object, image_object_keypoints,
		cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::drawKeypoints(image_scene, keypoints_scene, image_scene_keypoints,
		cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	// 显示特征点
	//cv::namedWindow("KeyPoints of image_object", cv::WINDOW_NORMAL);
	//cv::namedWindow("KeyPoints of image_scene", cv::WINDOW_NORMAL);
	//cv::imshow("KeyPoints of image_object", image_object_keypoints);
	//cv::imshow("KeyPoints of image_scene", image_scene_keypoints);

	//cv::imwrite("D:\\ProjectD\\test_result\\image_object_keypoints1.jpg", image_object_keypoints);
	//cv::imwrite("D:\\ProjectD\\test_result\\image_scene_keypoints1.jpg", image_scene_keypoints);

	// 特征点匹配，将计算结果保存在descriptors_object，descriptors_scene中
	cv::Mat descriptors_object, descriptors_scene;

	// 提取特征点并计算特征描述子
	detector->detectAndCompute(image_object, cv::Mat(),
		keypoints_object, descriptors_object);
	detector->detectAndCompute(image_scene, cv::Mat(),
		keypoints_scene, descriptors_scene);

	//-- Step 2: Matching descriptor vectors with a FLANN based matcher
	// 如果采用flannBased方法 那么 desp通过orb的到的类型不同需要先转换类型
	if (descriptors_object.type() != CV_32F || descriptors_scene.type() != CV_32F)
	{
		descriptors_object.convertTo(descriptors_object, CV_32F);
		descriptors_scene.convertTo(descriptors_scene, CV_32F);
	}

	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
	// DMatch主要用来储存匹配信息的结构体
	// query是要匹配的描述子，train是被匹配的描述子
	// 在Opencv中进行匹配时，void DescriptorMatcher::match(const Mat& queryDescriptors, const Mat&trainDescriptors, vector<DMatch>& matches, const Mat& mask) const
	// match函数的参数中位置在前面的为query descriptor，后面的是 train descriptor
	// 例如：query descriptor（mask）的数目为20，train descriptor（待匹配图像）数目为30，则DescriptorMatcher::match后的vector<DMatch>的size为20，若反过来，则vector<DMatch>的size为30
	// 简单来说DescriptorMatcher::match后的vector<DMatch>的size就是image_object中特征点的个数
	std::vector<std::vector<cv::DMatch>> knn_matches;

	// knn_matches为二维数组
	// 第一个维度的大小等于image_object中关键点的数目
	// 第二个维度的大小等于2，即从image_scene选出的候选点的个数
	// knn_matches中存储的是从image_scene选出的候选点的index
	matcher->knnMatch(descriptors_object, descriptors_scene, knn_matches, 2);

	//--filter matches using the Lowe's ratio test
	// 对于image_object中的任意一个关键点
	// 筛选出scene_image中与其距离最近的两个候选点，如果这两个候选点的距离很相近
	// 那么则将其在image_object中对应关键点剔除，因为其很容易引入错误的匹配
	const float ratio_thresh = 0.6f;
	std::vector<cv::DMatch> good_matches;
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		// 打印两个候选点的距离看一下
		//std::cout << "the matched keypoints's diatance" << std::endl;
		//std::cout << knn_matches[i][0].distance << "; " << knn_matches[i][1].distance << std::endl;

		if (knn_matches[i][0].distance <
			ratio_thresh * knn_matches[i][1].distance)
		{
			good_matches.push_back(knn_matches[i][0]);
		}
	}

	//--Draw matched
	cv::Mat img_matches;
	cv::drawMatches(image_object, keypoints_object, image_scene, keypoints_scene,
		good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
		std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//cv::namedWindow("picture of matching with ratio_thresh", cv::WINDOW_NORMAL);
	//cv::imshow("picture of matching with ratio_thresh", img_matches);


	//--删除错误匹配的特征点,将最优点放进inliers
	// 定义内点集合
	// 把keypoint转换为Point2f格式
	// Point(x, y) is using (x, y) as (column, row)；在用点的时候是（列，行）
	// src.at(i, j) is using (i, j) as (row, column)；也就是（行，列）
	std::vector<cv::DMatch> inliers;
	std::vector<cv::Point2f> object_keypoints_ransac;
	std::vector<cv::Point2f> scene_keypoints_ransac;

	for (size_t i = 0; i < good_matches.size(); i++)
	{
		// 把匹配后的点的坐标取出来存入object_keypoints_ransac、scene_keypoints_ransac
		object_keypoints_ransac.push_back(keypoints_object[good_matches[i].queryIdx].pt);
		scene_keypoints_ransac.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
	}

	/*for (size_t i = 0; i < good_matches.size(); i++)
	{
		std::cout << "the matched key_points's coordinate before RANSIC" << std::endl;
		std::cout << object_keypoints_ransac[i] << std::endl;
		std::cout << scene_keypoints_ransac[i] << std::endl;
	}*/

	//--RANSAC FindFundamental剔除错误点，这里传入的是成对的匹配点的坐标
	// 即query descriptor的坐标和与其匹配的train descriptor的坐标
	// 用以标记每一个匹配点的状态，等于0则为外点，等于1则为内点。
	std::vector<uchar> ransac_status;
	cv::findFundamentalMat(object_keypoints_ransac, scene_keypoints_ransac,
		ransac_status, cv::FM_RANSAC);

	for (size_t i = 0; i < good_matches.size(); i++)
	{
		if (ransac_status[i] != 0)
		{
			// 把经过RANSAC处理后的点的数据取出来存入inliers
			inliers.push_back(good_matches[i]);
		}
	}

	cv::Mat img_matches_ransac;
	cv::drawMatches(image_object, keypoints_object, image_scene, keypoints_scene,
		inliers, img_matches_ransac, cv::Scalar::all(-1), cv::Scalar::all(-1),
		std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//cv::namedWindow("picture of matching with RANSAC", cv::WINDOW_NORMAL);
	//cv::imshow("picture of matching with RANSAC", img_matches_ransac);

	for (size_t i = 0; i < inliers.size(); i++)
	{
		// 把匹配后的点取出来存入object_keypoints_ransac、scene_keypoints_ransac
		object_keypoints_ransac_inliers.push_back(keypoints_object[inliers[i].queryIdx].pt);
		scene_keypoints_ransac_inliers.push_back(keypoints_scene[inliers[i].trainIdx].pt);
	}

	return 0;
}

#endif // !AKAZE_H
