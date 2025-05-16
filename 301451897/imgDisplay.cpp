/*
  Yunyu Guo
  January 19 2025
  CS 5330 Project1-1
 */
#include <cstdio> // gives me printf
#include <cstring> // gives me strcpy
#include <opencv2/opencv.hpp> // openCV

int main(int argc, char *argv[]) {
  cv::Mat src;
  cv::Mat dst;
  char filename[256];// Buffer to store input filename


  // check for a command line argument
  if(argc < 2 ) {
    printf("usage: %s  <image filename>\n", argv[0]); // argv[0] is the program name
    exit(-1);
  }
  strncpy(filename, argv[1], 255); // Safely copy filename from command line

  src = cv::imread( filename ); // by default, returns image as 8-bit BGR image (if it's color), use IMREAD_UNCHANGED to keep the original data format
  if( src.data == NULL) { // no data, no image
    printf("error: unable to read image %s\n", filename);
    exit(-2);
  }

  cv::imshow( filename, src ); // display the original image

  // modify the image
  src.copyTo( dst ); // copy the src image to dst image

 
  
  // use the ptr<> method, this is much, much faster than at<> method but does the same channel swapping
  for(int i=0;i<dst.rows;i++) {
    cv::Vec3b *ptr = dst.ptr<cv::Vec3b>(i); // get the pointer for the row i data
    for(int j=0;j<dst.cols;j++) {
      unsigned char tmp = ptr[j][0];
      ptr[j][0] = ptr[j][2];
      ptr[j][2] = tmp;
    }
  }
 

  cv::imshow( "swapped", dst );

  // Enter loop to check for 'q' keypress or window closure
  while(cv::getWindowProperty("swapped", cv::WND_PROP_VISIBLE) >= 1) {//check to ensure the window is still visible
    char key = (char)cv::waitKey(10); // Wait for keypress with 10ms delay
    if(key == 'q' || key == 'Q' ) { 
      break;  
    }
  }

  cv::imwrite( "swap.png", dst ); // Save processed image

  printf("Terminating\n");
  
  return(0);
}

  
  

