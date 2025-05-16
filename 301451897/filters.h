/*
  Yunyu Guo
  CS 5330 Project1
  January 19 2025
*/

#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/opencv.hpp>

cv::Mat convertToGray(const cv::Mat& frame); //const: the input image won't be modified; cv::Mat& is a reference to a Matrix object (avoids copying large image data); frame: the parameter name for the input image 
int greyscale(cv::Mat &src, cv::Mat &dst);
int sepia(cv::Mat &src, cv::Mat &dst);  
int blur5x5_1(cv::Mat &src, cv::Mat &dst);
int blur5x5_2(cv::Mat &src, cv::Mat &dst);
int sobelX3x3(cv::Mat &src, cv::Mat &dst);
int sobelY3x3(cv::Mat &src, cv::Mat &dst);
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);

#endif 