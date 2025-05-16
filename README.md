# Project 1: Video-special effects


## Development Environment
- **Operating System:** [e.g., macOS Ventura 13.5, Windows 11 Pro, Ubuntu 22.04 LTS]
- **IDE/Compiler:** [e.g., Visual Studio Code with g++ 11.2, XCode 14.3, Terminal with Make & Clang, CLion with MinGW]
- **Key Libraries:**
    - OpenCV Version: OpenCV 4.x (as specified)
    - ONNX Runtime: [e.g., CPU version 1.1x.x, or specify if GPU version used]


## Project Overview
This project focuses on familiarization with C/C++, the OpenCV library, and image/video manipulation techniques. Tasks include reading and displaying images, capturing and processing live video, and implementing various image filters such as greyscale conversion, sepia tone, blur, Sobel edge detection, and quantization. A key new feature involves integrating the Depth Anything V2 deep learning network via ONNX Runtime for depth estimation and creating depth-based visual effects.

## Files Submitted
- `imgDisplay.cpp`
- `vidDisplay.cpp`
- `filters.cpp`
- `filters.h`
- `[YourProjectName]_Report.pdf`
- `readme.md`
- `Makefile` (if used)
- `DA2Network.hpp` (if modified or as provided for Task 11)
- Potentially other `.h` or `.cpp` files if you structured your project with more (e.g., `faceDetect.cpp`, `faceDetect.h`)


### Compilation
- If using a Makefile: `make`

### Execution
- **Image Display:** `./bin/imgDisplay path/to/your/image.jpg`
- **Video Special Effects:** `./bin/vidDisplay`

### Key Bindings for `vidDisplay`
- **q:** Quit the program.
- **s:** Save the current frame to an image file (e.g., `saved_frame.png`).
- **g:** Toggle OpenCV's standard greyscale filter.
- **h:** Toggle custom alternative greyscale filter.
- **b:** Toggle 5x5 blur filter (your implementation).
- **x:** Show X Sobel filter output (absolute values).
- **y:** Show Y Sobel filter output (absolute values).
- **m:** Show gradient magnitude image.
- **l:** Toggle blur and quantize filter.
- **f:** Toggle face detection.
- **d:** Toggle Depth Anything V2 based effect (or specify the key you used for Task 11).


