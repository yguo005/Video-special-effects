/*
  Yunyu Guo
  CS 5330 Project1
  January 19 2025
*/

#include <cstdio>
#include <cstring>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "DA2Network.hpp"
#include "faceDetect.h"
#include "showFaces.h"
#include "filters.h"

// opens a video stream and runs it through the depth anything network
// displays both the original video stream and the depth stream
int main(int argc, char *argv[]) {
  cv::VideoCapture *capdev;
  cv::Mat src; 
  cv::Mat dst;
  cv::Mat dst_vis;
  char filename[256]; // a string for the filename
  const float reduction = 0.5;

  // Initialize face detection
  std::string cascadePath = "haarcascade_frontalface_alt2.xml";
  cv::CascadeClassifier cascade;//Creates a CascadeClassifier object that will load and use the face detection model
  if (!cascade.load(cascadePath)) {
    printf("Could not load cascade classifier\n");
    return(-1);
  }

  // make a DANetwork object
  DA2Network da_net( "../model_fp16.onnx" );

  // open the video device
  capdev = new cv::VideoCapture(0);
  if( !capdev->isOpened() ) {
    printf("Unable to open video device\n");
    return(-1);
  }

  cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
		 (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));

  printf("Expected size: %d %d\n", refS.width, refS.height);

  float scale_factor = 256.0 / (refS.height*reduction);
  printf("Using scale factor %.2f\n", scale_factor);

  cv::namedWindow( "Video", 1 );
  cv::namedWindow( "Depth", 2 );

  for(;;) {
    // capture the next frame
    *capdev >> src;
    if( src.empty()) {
      printf("frame is empty\n");
      break;
    }
    
    // for speed purposes, reduce the size of the input frame by half
    cv::resize( src, src, cv::Size(), reduction, reduction );

    // Detect faces
    std::vector<cv::Rect> faces;
    detectFaces(src, faces);

    // Draw faces on the source image
    showFaces(src, faces);

    // Create a copy for processing
    cv::Mat processed_src = src.clone();
    cv::Mat display_image = src.clone();

    // Apply effects based on keyboard input
    static bool useBlur = false;
    static bool useLocalBlur = false;
    static bool useFog = false;
    static bool useColorFace = false;  

    if (useColorFace && !faces.empty()) {
        // Create a greyscale version of the entire image
        cv::Mat grey;
        greyscale(src, display_image);  
        // Create a mask for faces (black faces, white background)
        cv::Mat mask = cv::Mat::ones(src.size(), CV_8UC1) * 255;
        for (const auto& face : faces) {
            cv::rectangle(mask, face, cv::Scalar(0), -1);
        }

        // Copy original colored faces onto greyscale image
        src.copyTo(display_image, ~mask);  // Inverted mask to copy faces
    }

    if (useBlur) {
        // Apply global blur
        blur5x5_2(src, processed_src);
    } else if (useLocalBlur && !faces.empty()) {
        // Create a white mask (255) the same size as the source image
        cv::Mat mask = cv::Mat::ones(src.size(), CV_8UC1) * 255;
        for (const auto& face : faces) {//For each detected face
            cv::rectangle(mask, face, cv::Scalar(0), -1); // draws a filled black rectangle (0) where the face is, -1 fills the rectangle
        }
        
        // Create blurred version of the entire image
        cv::Mat blurred;
        blur5x5_2(src, blurred);
        
        // Copy original faces to the blurred image
        //copyTo(dst, src, mask): Where mask is NON-ZERO (typically 255/white): copies pixels from source to destination; Where mask is ZERO (0/black): leaves destination pixels unchanged
        src.copyTo(blurred, ~mask);  // copyTo copies pixels from src to blurred only where the inverted mask is white; ~ inverts the maskblack becomes white, white becomes black)
        processed_src = blurred;
    }

    // set the network input before running it
    da_net.set_input(processed_src, scale_factor);
    
    // run the network to get depth
    da_net.run_network(dst, src.size());//Outputs a depth map (dst) where each pixel value represents distance

    if (useFog) {
        // Normalize depth values to 0-1 range for fog intensity
        cv::Mat depth_normalized;
        cv::normalize(dst, depth_normalized, 0, 1, cv::NORM_MINMAX);
        
        // Blend original image with fog based on depth
        //If fog_intensity = 0 (close object):pixel = pixel * 1 + 255 * 0 = original color. If fog_intensity = 1 (far object):pixel = pixel * 0 + 255 * 1 = white (fog)
        for(int i = 0; i < src.rows; i++) {
            for(int j = 0; j < src.cols; j++) {
                float fog_intensity = depth_normalized.at<float>(i, j);
                cv::Vec3b &pixel = display_image.at<cv::Vec3b>(i, j);
                pixel[0] = pixel[0] * (1 - fog_intensity) + 255 * fog_intensity; // B
                pixel[1] = pixel[1] * (1 - fog_intensity) + 255 * fog_intensity; // G
                pixel[2] = pixel[2] * (1 - fog_intensity) + 255 * fog_intensity; // R
            }
        }
    }

    // apply a color map to the depth output to get a good visualization
    cv::applyColorMap(dst, dst_vis, cv::COLORMAP_INFERNO);

    // display the images
    cv::imshow("Video", display_image);  // Shows original + any effects (blur, fog)
    cv::imshow("Depth", dst_vis);        // Shows only depth visualization

    // handle key inputs
    static int imageCount = 0;  // Add counter for image filenames

    char key = cv::waitKey(10);
    if (key == 'q') {
        break;
    } else if (key == 's') {
        std::string filename = "saved_image_" + std::to_string(imageCount++) + ".jpg";
        
        // Save the processed image with current effects
        cv::imwrite(filename, display_image);
        
        printf("Image saved as: %s (Effects: %s%s%s%s)\n", 
            filename.c_str(),
            useBlur ? "Blur " : "",
            useLocalBlur ? "Local-Blur " : "",
            useFog ? "Fog " : "",
            useColorFace ? "Color-Face " : "");
            
    } else if (key == 'b') {
        useBlur = !useBlur;  // Toggle global blur
        useLocalBlur = false;
        useFog = false;
        useColorFace = false;
        printf("Global blur: %s\n", useBlur ? "ON" : "OFF");
    } else if (key == 'l') {
        useLocalBlur = !useLocalBlur;  // Toggle local blur
        useBlur = false;
        useFog = false;
        useColorFace = false;
        printf("Local blur: %s\n", useLocalBlur ? "ON" : "OFF");
    } else if (key == 'f') {
        useFog = !useFog;  // Toggle fog effect
        useBlur = false;
        useLocalBlur = false;
        useColorFace = false;
        printf("Fog effect: %s\n", useFog ? "ON" : "OFF");
    } else if (key == 'c') {  // Toggle color face effect
        useColorFace = !useColorFace;
        useBlur = false;
        useLocalBlur = false;
        useFog = false;
        printf("Color face effect: %s\n", useColorFace ? "ON" : "OFF");
    }
  }

  printf("Terminating\n");

  return(0);
}

