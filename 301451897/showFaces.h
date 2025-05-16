/*
  Yunyu Guo
  CS 5330 Project1
  January 19 2025
*/
#ifndef SHOWFACES_H
#define SHOWFACES_H

#include <opencv2/opencv.hpp>
#include <vector>

void showFaces(cv::Mat &frame, const std::vector<cv::Rect> &faces);

#endif