// realsense_color_view.cpp
// Build:
// g++ -std=c++17 realsense_color_view.cpp -o realsense_color_view `pkg-config --cflags --libs opencv4` -lrealsense2

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

int main() try {
    // Configure depth and color streams
    rs2::pipeline pipeline;
    rs2::config   config;

    // Resolve to inspect device info before starting the pipeline
    rs2::pipeline_wrapper wrapper(pipeline);
    rs2::pipeline_profile profile = config.resolve(wrapper);
    rs2::device device = profile.get_device();
    std::string product_line = device.get_info(RS2_CAMERA_INFO_PRODUCT_LINE);

    // Look for an RGB sensor
    bool found_rgb = false;
    for (auto&& s : device.query_sensors()) {
        if (s.supports(RS2_CAMERA_INFO_NAME)) {
            if (std::string(s.get_info(RS2_CAMERA_INFO_NAME)) == "RGB Camera") {
                found_rgb = true;
                break;
            }
        }
    }
    if (!found_rgb) {
        std::cerr << "The demo requires a depth camera with a Color sensor\n";
        return 1;
    }

    // Enable streams (depth + color). We’ll only display color (like your Python).
    config.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    if (product_line == "L500") {
        config.enable_stream(RS2_STREAM_COLOR, 960, 540, RS2_FORMAT_BGR8, 30);
    } else {
        config.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    }

    // Start streaming
    pipeline.start(config);

    cv::namedWindow("RealSense", cv::WINDOW_AUTOSIZE);

    while (true) {
        // Wait for a coherent pair (depth + color). We’ll just use color here.
        rs2::frameset frames = pipeline.wait_for_frames();
        // rs2::depth_frame depth = frames.get_depth_frame();
        rs2::video_frame color = frames.get_color_frame();
        if (!color) continue;

        // Convert to OpenCV Mat (BGR8)
        const int w = color.get_width();
        const int h = color.get_height();
        cv::Mat color_image(cv::Size(w, h), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);

        // Show image
        cv::imshow("RealSense", color_image);
        // Exit on ESC
        int key = cv::waitKey(1);
        if ((key & 0xFF) == 27) break;
    }

    // Stop streaming
    pipeline.stop();
    return 0;

} catch (const rs2::error& e) {
    std::cerr << "RealSense error calling " << e.get_failed_function()
              << "(" << e.get_failed_args() << "): " << e.what() << "\n";
    return 2;
} catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return 3;
}

