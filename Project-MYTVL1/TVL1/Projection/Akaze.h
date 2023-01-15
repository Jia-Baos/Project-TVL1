#pragma once
#ifndef AKAZE_H
#define AKAZE_H

#include <opencv2/opencv.hpp>


//��ֵ����+RANSAC
bool AKazeRegistration(const cv::Mat image_object, const cv::Mat image_scene,
	std::vector<cv::Point2f>& object_keypoints_ransac_inliers, std::vector<cv::Point2f>& scene_keypoints_ransac_inliers)
{
	//-- Step 1: Detect the keypoints using AKaze Detector, compute the descriptors
	//��ȡ�����㷽��
	cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();

	// ������
	// angle���Ƕȣ���ʾ�ؼ���ķ���
	// class_id����Ҫ��ͼƬ���з���ʱ�����ǿ�����class_id��ÿ��������������֣�δ�趨ʱΪ-1����Ҫ���Լ��趨
	// octave����ʾ�ؼ������Խ���������һ��
	// pt���ؼ��������
	// response���ؼ������Ӧ�̶�
	// size���ؼ���ֱ���Ĵ�С
	std::vector<cv::KeyPoint> keypoints_object, keypoints_scene;

	// ������ȡ������
	// ���������������ֱ����keypoints_object, keypoints_scene��
	detector->detect(image_object, keypoints_object);
	detector->detect(image_scene, keypoints_scene);

	// ����Mat�����������ƹؼ���
	cv::Mat image_object_keypoints;
	cv::Mat image_scene_keypoints;
	cv::drawKeypoints(image_object, keypoints_object, image_object_keypoints,
		cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::drawKeypoints(image_scene, keypoints_scene, image_scene_keypoints,
		cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	// ��ʾ������
	//cv::namedWindow("KeyPoints of image_object", cv::WINDOW_NORMAL);
	//cv::namedWindow("KeyPoints of image_scene", cv::WINDOW_NORMAL);
	//cv::imshow("KeyPoints of image_object", image_object_keypoints);
	//cv::imshow("KeyPoints of image_scene", image_scene_keypoints);

	//cv::imwrite("D:\\ProjectD\\test_result\\image_object_keypoints1.jpg", image_object_keypoints);
	//cv::imwrite("D:\\ProjectD\\test_result\\image_scene_keypoints1.jpg", image_scene_keypoints);

	// ������ƥ�䣬��������������descriptors_object��descriptors_scene��
	cv::Mat descriptors_object, descriptors_scene;

	// ��ȡ�����㲢��������������
	detector->detectAndCompute(image_object, cv::Mat(),
		keypoints_object, descriptors_object);
	detector->detectAndCompute(image_scene, cv::Mat(),
		keypoints_scene, descriptors_scene);

	//-- Step 2: Matching descriptor vectors with a FLANN based matcher
	// �������flannBased���� ��ô despͨ��orb�ĵ������Ͳ�ͬ��Ҫ��ת������
	if (descriptors_object.type() != CV_32F || descriptors_scene.type() != CV_32F)
	{
		descriptors_object.convertTo(descriptors_object, CV_32F);
		descriptors_scene.convertTo(descriptors_scene, CV_32F);
	}

	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
	// DMatch��Ҫ��������ƥ����Ϣ�Ľṹ��
	// query��Ҫƥ��������ӣ�train�Ǳ�ƥ���������
	// ��Opencv�н���ƥ��ʱ��void DescriptorMatcher::match(const Mat& queryDescriptors, const Mat&trainDescriptors, vector<DMatch>& matches, const Mat& mask) const
	// match�����Ĳ�����λ����ǰ���Ϊquery descriptor��������� train descriptor
	// ���磺query descriptor��mask������ĿΪ20��train descriptor����ƥ��ͼ����ĿΪ30����DescriptorMatcher::match���vector<DMatch>��sizeΪ20��������������vector<DMatch>��sizeΪ30
	// ����˵DescriptorMatcher::match���vector<DMatch>��size����image_object��������ĸ���
	std::vector<std::vector<cv::DMatch>> knn_matches;

	// knn_matchesΪ��ά����
	// ��һ��ά�ȵĴ�С����image_object�йؼ������Ŀ
	// �ڶ���ά�ȵĴ�С����2������image_sceneѡ���ĺ�ѡ��ĸ���
	// knn_matches�д洢���Ǵ�image_sceneѡ���ĺ�ѡ���index
	matcher->knnMatch(descriptors_object, descriptors_scene, knn_matches, 2);

	//--filter matches using the Lowe's ratio test
	// ����image_object�е�����һ���ؼ���
	// ɸѡ��scene_image��������������������ѡ�㣬�����������ѡ��ľ�������
	// ��ô������image_object�ж�Ӧ�ؼ����޳�����Ϊ���������������ƥ��
	const float ratio_thresh = 0.6f;
	std::vector<cv::DMatch> good_matches;
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		// ��ӡ������ѡ��ľ��뿴һ��
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


	//--ɾ������ƥ���������,�����ŵ�Ž�inliers
	// �����ڵ㼯��
	// ��keypointת��ΪPoint2f��ʽ
	// Point(x, y) is using (x, y) as (column, row)�����õ��ʱ���ǣ��У��У�
	// src.at(i, j) is using (i, j) as (row, column)��Ҳ���ǣ��У��У�
	std::vector<cv::DMatch> inliers;
	std::vector<cv::Point2f> object_keypoints_ransac;
	std::vector<cv::Point2f> scene_keypoints_ransac;

	for (size_t i = 0; i < good_matches.size(); i++)
	{
		// ��ƥ���ĵ������ȡ��������object_keypoints_ransac��scene_keypoints_ransac
		object_keypoints_ransac.push_back(keypoints_object[good_matches[i].queryIdx].pt);
		scene_keypoints_ransac.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
	}

	/*for (size_t i = 0; i < good_matches.size(); i++)
	{
		std::cout << "the matched key_points's coordinate before RANSIC" << std::endl;
		std::cout << object_keypoints_ransac[i] << std::endl;
		std::cout << scene_keypoints_ransac[i] << std::endl;
	}*/

	//--RANSAC FindFundamental�޳�����㣬���ﴫ����ǳɶԵ�ƥ��������
	// ��query descriptor�����������ƥ���train descriptor������
	// ���Ա��ÿһ��ƥ����״̬������0��Ϊ��㣬����1��Ϊ�ڵ㡣
	std::vector<uchar> ransac_status;
	cv::findFundamentalMat(object_keypoints_ransac, scene_keypoints_ransac,
		ransac_status, cv::FM_RANSAC);

	for (size_t i = 0; i < good_matches.size(); i++)
	{
		if (ransac_status[i] != 0)
		{
			// �Ѿ���RANSAC�����ĵ������ȡ��������inliers
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
		// ��ƥ���ĵ�ȡ��������object_keypoints_ransac��scene_keypoints_ransac
		object_keypoints_ransac_inliers.push_back(keypoints_object[inliers[i].queryIdx].pt);
		scene_keypoints_ransac_inliers.push_back(keypoints_scene[inliers[i].trainIdx].pt);
	}

	return 0;
}

#endif // !AKAZE_H
