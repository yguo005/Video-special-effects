/*
  Yunyu Guo
  January 19 2025
  CS 5330 Project1-2
 */
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "filters.h"
#include "faceDetect.h"


enum DisplayMode {
    COLOR,
    STANDARD_GRAY,
    CUSTOM_GRAY,
    SEPIA
};

int main(int argc, char *argv[]) {
cv::VideoCapture *capdev;
// open the video device
capdev = new cv::VideoCapture(0);
if( !capdev->isOpened() ) {
printf("Unable to open video device\n");
return(-1);
}
// get some properties of the image
cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
(int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
printf("Expected size: %d %d\n", refS.width, refS.height);
cv::namedWindow("Video", 1); // identifies a window
cv::Mat frame;


int imageCount = 0;

DisplayMode displayMode = COLOR;
bool blurEnabled = false;
bool sobelXEnabled = false;
bool sobelYEnabled = false;
bool magnitudeEnabled = false;
bool quantizeEnabled = false;
const int QUANTIZE_LEVELS = 10;
bool faceDetectEnabled = false;
std::vector<cv::Rect> faces;// Stores detected faces as rectangles
cv::Rect last(0, 0, 0, 0);// Stores previous frame's face position
cv::Mat grey;//face detection works on grayscale

for(;;) {//Infinite loop to continuously capture and display frames
    *capdev >> frame; // Captures a new frame from the camera and stores it in the frame variable
    if( frame.empty() ) {
        printf("frame is empty\n");
        break;
    }

    // Create a Matrix object to store the processed frame
    cv::Mat displayFrame;
    switch(displayMode) {
        case STANDARD_GRAY:
            displayFrame = convertToGray(frame);
            break;
        case CUSTOM_GRAY:
            displayFrame = frame.clone();
            greyscale(frame, displayFrame);
            break;
        case SEPIA:
            displayFrame = frame.clone();
            sepia(frame, displayFrame);
            break;
        default: // COLOR
            displayFrame = frame.clone();
            break;
    }

    // Apply blur if enabled (using optimized blur5x5_2)
    if(blurEnabled) {
        cv::Mat blurredFrame;
        blur5x5_2(displayFrame, blurredFrame);
        displayFrame = blurredFrame;
    }

    if(sobelXEnabled || sobelYEnabled) {
        cv::Mat sobelOut;
        cv::Mat displaySobel;
        
        if(sobelXEnabled) {
            sobelX3x3(displayFrame, sobelOut);
        } else {
            sobelY3x3(displayFrame, sobelOut);
        }
        
        // Convert to displayable format
        cv::convertScaleAbs(sobelOut, displaySobel);//convertScaleAbs: Makes sobelOut (signed short) displayable displaySobel (unsigned char)
        displayFrame = displaySobel;//Updates what we show
    }

    if(magnitudeEnabled) {
        cv::Mat sobelX, sobelY, magOut;
        
        // Calculate both Sobel derivatives
        sobelX3x3(displayFrame, sobelX);
        sobelY3x3(displayFrame, sobelY);
        
        // Calculate magnitude
        magnitude(sobelX, sobelY, magOut);
        
        displayFrame = magOut;
    }

    if(quantizeEnabled) {
        cv::Mat quantizedFrame;
        blurQuantize(displayFrame, quantizedFrame, QUANTIZE_LEVELS);
        displayFrame = quantizedFrame;
    }

    if(faceDetectEnabled) {
        // Convert to grayscale for face detection
        cv::cvtColor(displayFrame, grey, cv::COLOR_BGR2GRAY, 0);
        
        // Detect faces
        detectFaces(grey, faces);
        
        // Draw boxes
        drawBoxes(displayFrame, faces);
        
        // Smooth detection by averaging with last frame
        if(faces.size() > 0) {
            last.x = (faces[0].x + last.x)/2;
            last.y = (faces[0].y + last.y)/2;
            last.width = (faces[0].width + last.width)/2;
            last.height = (faces[0].height + last.height)/2;
        }
    }

    cv::imshow("Video", displayFrame);
    
    // if there is a waiting keystroke
    char key = cv::waitKey(10);
    if( key == 'q') {
        break;
    } else if(key == 's') {
        std::string filename = "saved_image_" + std::to_string(imageCount++) + ".jpg";
        
        // Apply the same processing as display
        cv::Mat saveFrame;
    
        switch(displayMode) {
            case STANDARD_GRAY:
                saveFrame = convertToGray(frame);
                break;
            case CUSTOM_GRAY:
                saveFrame = frame.clone();
                greyscale(frame, saveFrame);
                break;
            case SEPIA:
                saveFrame = frame.clone();
                sepia(frame, saveFrame);
                break;
            default: // COLOR
                saveFrame = frame.clone();
                break;
        }
        
        // Then apply blur if enabled
        if(blurEnabled) {
            cv::Mat blurredFrame;
            blur5x5_2(saveFrame, blurredFrame);
            saveFrame = blurredFrame;
        }
        
        // Apply magnitude if enabled
        if(magnitudeEnabled) {
            cv::Mat sobelX, sobelY, magOut;
            sobelX3x3(saveFrame, sobelX);
            sobelY3x3(saveFrame, sobelY);
            magnitude(sobelX, sobelY, magOut);
            saveFrame = magOut;
        }

        // Apply quantize if enabled
        if(quantizeEnabled) {
            cv::Mat quantizedFrame;
            blurQuantize(saveFrame, quantizedFrame, QUANTIZE_LEVELS);
            saveFrame = quantizedFrame;
        }
        // Draw boxes if face detection is enabled
        if(faceDetectEnabled && faces.size() > 0) {
            drawBoxes(saveFrame, faces);
        }

        // Save the processed image
        cv::imwrite(filename, saveFrame);
        printf("Image saved as: %s (Mode: %s, Blur: %s, Magnitude: %s, Quantize: %s)\n", 
            filename.c_str(),
            displayMode == COLOR ? "Color" : 
            displayMode == STANDARD_GRAY ? "Standard Grayscale" : 
            displayMode == CUSTOM_GRAY ? "Custom Grayscale" : 
            "Sepia",
            blurEnabled ? "ON" : "OFF",
            magnitudeEnabled ? "ON" : "OFF",
            quantizeEnabled ? "ON" : "OFF");
    } else if(key == 'g') {
        // Cycle through modes: COLOR -> STANDARD_GRAY -> CUSTOM_GRAY -> SEPIA -> COLOR
        displayMode = static_cast<DisplayMode>((displayMode + 1) % 4);  
        printf("Display mode: %s\n", 
            displayMode == COLOR ? "Color" : 
            displayMode == STANDARD_GRAY ? "Standard Grayscale" : 
            displayMode == CUSTOM_GRAY ? "Custom Grayscale" : 
            "Sepia");
    } else if(key == 'b') {
        blurEnabled = !blurEnabled;
        printf("Blur filter: %s\n", blurEnabled ? "ON" : "OFF");
    } else if(key == 'x') {
        sobelXEnabled = !sobelXEnabled;
        sobelYEnabled = false;
        printf("Sobel X filter: %s\n", sobelXEnabled ? "ON" : "OFF");
    } else if(key == 'y') {
        sobelYEnabled = !sobelYEnabled;
        sobelXEnabled = false;
        printf("Sobel Y filter: %s\n", sobelYEnabled ? "ON" : "OFF");
    } else if(key == 'm') {
        magnitudeEnabled = !magnitudeEnabled;
        sobelXEnabled = false;
        sobelYEnabled = false;
        printf("Magnitude display: %s\n", magnitudeEnabled ? "ON" : "OFF");
    } else if(key == 'l') {
        quantizeEnabled = !quantizeEnabled;
        printf("Blur Quantize: %s (Levels: %d)\n", 
               quantizeEnabled ? "ON" : "OFF", 
               QUANTIZE_LEVELS);
    } else if(key == 'f') {
        faceDetectEnabled = !faceDetectEnabled;
        printf("Face detection: %s\n", faceDetectEnabled ? "ON" : "OFF");
    }
} 

delete capdev;
return(0);
}
