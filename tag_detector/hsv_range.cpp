// hsv_threshold_trackbar.cpp
// Build (Ubuntu):
//   g++ -std=c++17 hsv_threshold_trackbar.cpp -o hsv_thresh `pkg-config --cflags --libs opencv4`
// Run:
//   ./hsv_thresh            # default camera 0
//   ./hsv_thresh 1          # camera 1

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

// Globals to mirror the Python variables
const int max_value     = 255;
const int max_value_H   = 360/2; // OpenCV Hue range: 0..179
int low_H  = 0,   high_H = max_value_H;
int low_S  = 0,   high_S = max_value;
int low_V  = 0,   high_V = max_value;

const std::string window_capture_name   = "Video Capture";
const std::string window_detection_name = "Object Detection";
const std::string low_H_name  = "Low H";
const std::string low_S_name  = "Low S";
const std::string low_V_name  = "Low V";
const std::string high_H_name = "High H";
const std::string high_S_name = "High S";
const std::string high_V_name = "High V";

// Callbacks (keep constraints: low < high)
static void on_low_H_thresh_trackbar(int val, void*) {
    low_H = std::min(val, high_H - 1);
    setTrackbarPos(low_H_name, window_detection_name, low_H);
}
static void on_high_H_thresh_trackbar(int val, void*) {
    high_H = std::max(val, low_H + 1);
    setTrackbarPos(high_H_name, window_detection_name, high_H);
}
static void on_low_S_thresh_trackbar(int val, void*) {
    low_S = std::min(val, high_S - 1);
    setTrackbarPos(low_S_name, window_detection_name, low_S);
}
static void on_high_S_thresh_trackbar(int val, void*) {
    high_S = std::max(val, low_S + 1);
    setTrackbarPos(high_S_name, window_detection_name, high_S);
}
static void on_low_V_thresh_trackbar(int val, void*) {
    low_V = std::min(val, high_V - 1);
    setTrackbarPos(low_V_name, window_detection_name, low_V);
}
static void on_high_V_thresh_trackbar(int val, void*) {
    high_V = std::max(val, low_V + 1);
    setTrackbarPos(high_V_name, window_detection_name, high_V);
}

int main(int argc, char** argv) {
    int cam_index = 0;
    if (argc >= 2) cam_index = std::atoi(argv[1]);

    VideoCapture cap(cam_index);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera " << cam_index << "\n";
        return 1;
    }

    namedWindow(window_capture_name);
    namedWindow(window_detection_name);

    createTrackbar(low_H_name,  window_detection_name, &low_H,  max_value_H, on_low_H_thresh_trackbar);
    createTrackbar(high_H_name, window_detection_name, &high_H, max_value_H, on_high_H_thresh_trackbar);
    createTrackbar(low_S_name,  window_detection_name, &low_S,  max_value,   on_low_S_thresh_trackbar);
    createTrackbar(high_S_name, window_detection_name, &high_S, max_value,   on_high_S_thresh_trackbar);
    createTrackbar(low_V_name,  window_detection_name, &low_V,  max_value,   on_low_V_thresh_trackbar);
    createTrackbar(high_V_name, window_detection_name, &high_V, max_value,   on_high_V_thresh_trackbar);

    Mat frame, frame_HSV, frame_threshold;

    for (;;) {
        if (!cap.read(frame) || frame.empty()) break;

        cvtColor(frame, frame_HSV, COLOR_BGR2HSV);
        inRange(frame_HSV,
                Scalar(low_H,  low_S,  low_V),
                Scalar(high_H, high_S, high_V),
                frame_threshold);

        imshow(window_capture_name, frame);
        imshow(window_detection_name, frame_threshold);

        int key = waitKey(30);
        if (key == 'q' || key == 27) break; // 'q' or ESC
    }

    destroyAllWindows();
    return 0;
}

