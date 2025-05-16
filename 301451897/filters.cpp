/*
  Yunyu Guo
  January 19 2025
  CS 5330 Project1-2
 */

#include "filters.h"

/*************************     COLOR_BGR2GRAY     **********************************************/
cv::Mat convertToGray(const cv::Mat& frame) {
    cv::Mat grayFrame;
    cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
    return grayFrame;
}

/*************************     greyscale filter      **********************************************/
int greyscale(cv::Mat &src, cv::Mat &dst) {
    // Check if image is empty
    if(src.empty()) {
        return -1;
    }

    // Create destination image of same size as source
    dst.create(src.size(), src.type());

    // Iterate through each pixel
    for(int i = 0; i < src.rows; i++) {
        for(int j = 0; j < src.cols; j++) {
            // Get pixel value
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j); // Vec3b is one pixel in the image, Each Vec3b holds BGR values for one pixel
            
            // Get individual channel values
            int blue = pixel[0];
            int green = pixel[1];
            int red = pixel[2];

            // Custom grayscale conversion:
            // Invert red, reduce green and blue influence
            int gray_value = (255 - red) * 0.4 + 
                           green * 0.5 + 
                           blue * 0.1;

            // Ensure value is in valid range
            gray_value = std::min(255, std::max(0, gray_value));

            // Set all channels to the same value for grayscale
            dst.at<cv::Vec3b>(i, j) = cv::Vec3b(gray_value, gray_value, gray_value);
        }
    }

    return 0;
}

/*************************      sepia filter      **********************************************/
int sepia(cv::Mat &src, cv::Mat &dst) {
    // Check if image is empty
    if(src.empty()) {
        return -1;
    }

    // Create destination image of same size as source
    dst.create(src.size(), src.type());

    // Sepia coefficients
    const float sepia_r[] = {0.393f, 0.769f, 0.189f};  // Red coefficients
    const float sepia_g[] = {0.349f, 0.686f, 0.168f};  // Green coefficients
    const float sepia_b[] = {0.272f, 0.534f, 0.131f};  // Blue coefficients

    /*************************    vignetting       **********************************************/
    // vignetting: Calculate center of image 
    float centerX = src.cols / 2.0f; //src.cols gives image width
    float centerY = src.rows / 2.0f;
    
    // vignetting: Calculate maximum distance from center to corner
    float maxDistance = sqrt(centerX * centerX + centerY * centerY);

    // Iterate through each pixel
    for(int i = 0; i < src.rows; i++) {
        for(int j = 0; j < src.cols; j++) {
            // Get pixel value
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
            
            // Get original BGR values
            float blue = pixel[0];
            float green = pixel[1];
            float red = pixel[2];

            // Calculate new values using original sepia coefficients
            int newBlue = std::min(255, (int)(blue * sepia_b[2] + // B * 0.131
                                            green * sepia_b[1] + // G * 0.534
                                            red * sepia_b[0])); // R * 0.272
            
            int newGreen = std::min(255, (int)(blue * sepia_g[2] + 
                                             green * sepia_g[1] + 
                                             red * sepia_g[0]));
            
            int newRed = std::min(255, (int)(blue * sepia_r[2] + 
                                           green * sepia_r[1] + 
                                           red * sepia_r[0]));

            // Calculate distance from center for a pixel
            float distX = j - centerX;
            float distY = i - centerY;
            float distance = sqrt(distX * distX + distY * distY);
            
            // normalize distance: distance / maxDistance creates value from 0 to 1
            //1.0f -: invert the effect, centre full brightness =1
            float vignette = 1.0f - pow(distance / maxDistance, 8);

            // Apply vignette to sepia values
            newBlue = std::max(0, std::min(255, (int)(newBlue * vignette)));
            newGreen = std::max(0, std::min(255, (int)(newGreen * vignette)));
            newRed = std::max(0, std::min(255, (int)(newRed * vignette)));

            // Set new values
            dst.at<cv::Vec3b>(i, j) = cv::Vec3b(newBlue, newGreen, newRed);
        }
    }

    return 0;
}

/*************************      5x5_1 blur filter     **********************************************/
int blur5x5_1(cv::Mat &src, cv::Mat &dst) {
    if(src.empty()) {
        return -1;
    }

    // Copy source image to destination
    src.copyTo(dst);

    // Gaussian kernel 5x5, odd-sized because Gaussian blur is radially symmetric, has center pixel
    const int kernel[5][5] = {
        {1, 2, 4, 2, 1}, // Row -2
        {2, 4, 8, 4, 2}, // Row -1
        {4, 8, 16, 8, 4}, // Row 0 (center)
        {2, 4, 8, 4, 2}, // Row 1
        {1, 2, 4, 2, 1} // Row 2
    };
    
    // Sum of all kernel values for normalization
    const int kernelSum = 100;  // Sum of all values in kernel

    // Process each pixel except border (2 pixels on each side)
    for(int i = 2; i < src.rows-2; i++) {
        for(int j = 2; j < src.cols-2; j++) {
            int sumB = 0, sumG = 0, sumR = 0;

            // Apply kernel to neighborhood
            for(int k = -2; k <= 2; k++) { // rows: -2, -1, 0, 1, 2
                for(int l = -2; l <= 2; l++) { // cols: -2, -1, 0, 1, 2
                    // Get pixel value
                    cv::Vec3b pixel = src.at<cv::Vec3b>(i + k, j + l);
                    
                    // Multiply pixel values by kernel weight
                    int weight = kernel[k+2][l+2]; //k and l range from -2 to +2, add 2 to convert to 0 to 4 (array indices)
                    sumB += pixel[0] * weight;
                    sumG += pixel[1] * weight;
                    sumR += pixel[2] * weight;
                }
            }

            // Normalize and set new pixel values
            dst.at<cv::Vec3b>(i, j) = cv::Vec3b(
                sumB / kernelSum,
                sumG / kernelSum,
                sumR / kernelSum
            );
        }
    }

    return 0;
}


/*************************      5x5_2 blur filter    **********************************************/
int blur5x5_2(cv::Mat &src, cv::Mat &dst) {
    if(src.empty()) {
        return -1;
    }

    // Create temporary matrix for intermediate results
    //why need temp matrix: Allows independent horizontal and vertical passes
    cv::Mat temp = src.clone(); //Mat is the whole image (container),A color Mat is made up of many Vec3b pixels
    dst = src.clone();

    // 1D Gaussian kernel [1 2 4 2 1]
    const int kernel[] = {1, 2, 4, 2, 1};
    const int kernelSum = 10;  // Sum of kernel values

    // Horizontal pass
    for(int i = 0; i < src.rows; i++) {
        // Get pointers to rows
        uchar* tempRow = temp.ptr<uchar>(i); // Get pointer to row i in temp matrix
        const uchar* srcRow = src.ptr<uchar>(i); // Get pointer to row i in src matrix

        // Process each pixel except borders
        for(int j = 2; j < src.cols-2; j++) {// restrict horizontal movement
            // Process each channel
            for(int c = 0; c < 3; c++) {
                int sum = 0;
                // Apply horizontal kernel
                for(int k = -2; k <= 2; k++) {
                    sum += srcRow[3*(j+k) + c] * kernel[k+2]; //example: pixel 5, k = -2,  3*(5-2) Look 2 pixels left, Multiply by 3 to skip to correct pixel, add c to skip to correct channel
                    //kernel[k+2]: weight
                }
                tempRow[3*j + c] = sum / kernelSum; //normalize by kernelSum
            }
        }
    }

    // Vertical pass
    for(int i = 2; i < src.rows-2; i++) { // Moves down each column
        // Get pointer to current row
        uchar* dstRow = dst.ptr<uchar>(i);

        // Process each pixel except borders
        for(int j = 0; j < src.cols; j++) {// only need to restrict vertical movement
            // Process each channel
            for(int c = 0; c < 3; c++) {
                int sum = 0;
                // Apply vertical kernel
                for(int k = -2; k <= 2; k++) {
                    sum += temp.ptr<uchar>(i+k)[3*j + c] * kernel[k+2]; //temp.ptr<uchar>(i+k): Get row pointer, [3*j + c]: Get specific channel in that row, * kernel[k+2]: weight
                }
                dstRow[3*j + c] = sum / kernelSum;
            }
        }
    }

    return 0;
}

/*************************      3x3 sobelX filter     **********************************************/
//Shows only horizontal changes
int sobelX3x3(cv::Mat &src, cv::Mat &dst) {
    if(src.empty()) {
        return -1;
    }

    // Create destination matrix of type CV_16SC3
    dst.create(src.size(), CV_16SC3);
    
    // Create temporary matrix for intermediate results
    cv::Mat temp(src.size(), CV_16SC3);

    // Separable Sobel X kernels
    // Horizontal: [1 0 -1]
    // Vertical: [1; 2; 1]
    
    // First pass - vertical smoothing
    for(int i = 1; i < src.rows-1; i++) {// For each row except first and last
        const uchar* prev = src.ptr<uchar>(i-1);// Row above; uchar* for input (0-255 values)
        const uchar* curr = src.ptr<uchar>(i); // Current row
        const uchar* next = src.ptr<uchar>(i+1); // Row below
        short* tempRow = temp.ptr<short>(i); // output row; short* for output

        //Need 3 complete rows at once
        for(int j = 0; j < src.cols; j++) {// For each pixel in row
            for(int c = 0; c < 3; c++) {// For each channel
                tempRow[j*3 + c] = (short)(//store result in tempRow
                    prev[j*3 + c] + // above pixel *1
                    2 * curr[j*3 + c] + // current pixel *2
                    next[j*3 + c] // below pixel *1
                );
            }
        }
    }

    // Second pass - horizontal differentiation
    for(int i = 0; i < src.rows; i++) {
        short* tempRow = temp.ptr<short>(i);// Get pointer to temp row (input)
        short* dstRow = dst.ptr<short>(i); // Get pointer to dst row (output)

        //Need only left/right neighbors within same row
        for(int j = 1; j < src.cols-1; j++) {// Skip first/last columns
            for(int c = 0; c < 3; c++) {
                dstRow[j*3 + c] = 
                    tempRow[(j+1)*3 + c] - // Right pixel
                    tempRow[(j-1)*3 + c]; // Left pixel 
                    //right - left detects left-to-right change
            }
        }
    }

    return 0;
}

/*************************      3x3 sobelY filter     **********************************************/
// Shows only vertical changes
int sobelY3x3(cv::Mat &src, cv::Mat &dst) {
    if(src.empty()) {
        return -1;
    }

    // Create destination matrix of type CV_16SC3
    dst.create(src.size(), CV_16SC3);
    
    // Create temporary matrix for intermediate results
    cv::Mat temp(src.size(), CV_16SC3);

    // Separable Sobel Y kernels
    // Horizontal: [1 2 1]
    // Vertical: [1; 0; -1]
    
    // First pass - horizontal smoothing
    for(int i = 0; i < src.rows; i++) {
        const uchar* srcRow = src.ptr<uchar>(i);
        short* tempRow = temp.ptr<short>(i);

        for(int j = 1; j < src.cols-1; j++) {
            for(int c = 0; c < 3; c++) {
                tempRow[j*3 + c] = (short)(
                    srcRow[(j-1)*3 + c] +
                    2 * srcRow[j*3 + c] +
                    srcRow[(j+1)*3 + c]
                );
            }
        }
    }

    // Second pass - vertical differentiation
    for(int i = 1; i < src.rows-1; i++) {
        short* prevRow = temp.ptr<short>(i-1);
        short* nextRow = temp.ptr<short>(i+1);
        short* dstRow = dst.ptr<short>(i);

        for(int j = 0; j < src.cols; j++) {
            for(int c = 0; c < 3; c++) {
                dstRow[j*3 + c] = 
                    prevRow[j*3 + c] -
                    nextRow[j*3 + c];
            }
        }
    }

    return 0;
}

/*************************       magnitude filter     **********************************************/
//Combines horizontal and vertical changes
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst) {
    if(sx.empty() || sy.empty() || sx.size() != sy.size()) {
        return -1;
    }

    // Create destination matrix (CV_8UC3 for display)
    dst.create(sx.size(), CV_8UC3);

    for(int i = 0; i < sx.rows; i++) {
        // Get row pointers
        short* sx_row = sx.ptr<short>(i); //sx_row (Sobel X results)
        short* sy_row = sy.ptr<short>(i); //sy_row (Sobel Y results)
        uchar* dst_row = dst.ptr<uchar>(i); //dst_row (magnitude results)

        for(int j = 0; j < sx.cols; j++) {
            for(int c = 0; c < 3; c++) {
                // Get Sobel X and Y values
                float sx_val = sx_row[j*3 + c];
                float sy_val = sy_row[j*3 + c];

                // Calculate magnitude using Euclidean distance
                float mag = sqrt(sx_val*sx_val + sy_val*sy_val);

                // Scale to 0-255 range and store
                dst_row[j*3 + c] = cv::saturate_cast<uchar>(mag);
            }
        }
    }

    return 0;
}

/*************************       blurQuantize filter     **********************************************/
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels) {
    if(src.empty() || levels < 2) {
        return -1;
    }

    // First blur the image using existing blur5x5_2
    cv::Mat blurred;
    blur5x5_2(src, blurred);

    // Create destination matrix
    dst.create(src.size(), CV_8UC3); //Color BGR (3 channels) 0-255 range per channel

    // Calculate bucket size
    float bucket = 255.0f / (levels - 1); //Why (levels - 1): example: For 4 levels, need 3 intervals:

    for(int i = 0; i < src.rows; i++) {
        uchar* dst_row = dst.ptr<uchar>(i);
        uchar* blur_row = blurred.ptr<uchar>(i);

        for(int j = 0; j < src.cols; j++) {
            for(int c = 0; c < 3; c++) {
                // Get blurred value
                float val = blur_row[j*3 + c];
                
                // Quantize: divide by bucket size and round
                int temp = round(val / bucket);
                
                // Scale back to original range
                dst_row[j*3 + c] = (uchar)(temp * bucket);
            }
        }
    }

    return 0;
}