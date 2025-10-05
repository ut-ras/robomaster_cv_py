// depth_calculation.cpp
// Build:
//   g++ -std=c++17 depth_calculation.cpp -o depth_calc `pkg-config --cflags --libs opencv4` -lrealsense2

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <stdexcept>

// ---------- Globals (like your Python module-level singletons) ----------
static rs2::pipeline g_pipeline;
static rs2::config   g_config;
// Align depth to color (same as: align_to = rs.stream.color; align = rs.align(align_to))
static rs2::align    g_align(RS2_STREAM_COLOR);

// ---------- Initialize RealSense (equivalent to initialize_real_sense) ----------
void initialize_real_sense() {
    // Resolve first so we can inspect device before starting streaming
    rs2::pipeline_wrapper wrapper(g_pipeline);
    rs2::pipeline_profile pre_profile = g_config.resolve(wrapper);
    rs2::device dev = pre_profile.get_device();
    std::string product_line = dev.get_info(RS2_CAMERA_INFO_PRODUCT_LINE);

    // Ensure RGB camera exists
    bool found_rgb = false;
    for (auto&& s : dev.query_sensors()) {
        if (s.supports(RS2_CAMERA_INFO_NAME)) {
            if (std::string(s.get_info(RS2_CAMERA_INFO_NAME)) == "RGB Camera") {
                found_rgb = true;
                break;
            }
        }
    }
    if (!found_rgb) {
        std::cerr << "The demo requires Depth camera with Color sensor\n";
        std::exit(1);
    }

    // Depth stream 640x480@30 Z16
    g_config.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

    // Color stream 640x480@30 BGR8 (same for D435i and others here)
    g_config.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);

    // Start streaming
    g_pipeline.start(g_config);
}

// ---------- Get color & depth frames (aligned), as OpenCV Mats ----------
std::pair<cv::Mat, cv::Mat> get_color_depth_image() {
    try {
        rs2::frameset frames = g_pipeline.wait_for_frames();
        rs2::frameset aligned = g_align.process(frames);

        rs2::video_frame color = aligned.get_color_frame();
        rs2::depth_frame depth = aligned.get_depth_frame();
        if (!color || !depth) return {cv::Mat(), cv::Mat()};

        const int cw = color.get_width();
        const int ch = color.get_height();
        const int dw = depth.get_width();
        const int dh = depth.get_height();

        // color: BGR8 → CV_8UC3
        cv::Mat color_image(cv::Size(cw, ch), CV_8UC3,
                            (void*)color.get_data(), cv::Mat::AUTO_STEP);

        // depth: Z16 → CV_16UC1
        cv::Mat depth_image(cv::Size(dw, dh), CV_16UC1,
                            (void*)depth.get_data(), cv::Mat::AUTO_STEP);

        // Return shallow copies (valid until next wait_for_frames). If you need persistence, clone().
        return {color_image.clone(), depth_image.clone()};
    } catch (...) {
        try { g_pipeline.stop(); } catch (...) {}
        return {cv::Mat(), cv::Mat()};
    }
}

// ---------- BoundingBox interface expected (same getters/setters as your Python/C++) ----------
struct BoundingBox {
    // x1/x2/y1/y2 are pixel indices (inclusive/exclusive like Python slicing)
    int x1{0}, x2{0}, y1{0}, y2{0};
    float depth_m{0.f};
    void set_depth(float d) { depth_m = d; }

    // If you already have a C++ BoundingBox class with these methods, adapt here:
    std::pair<int,int> get_x_value() const { return {x1, x2}; }
    std::pair<int,int> get_y_value() const { return {y1, y2}; }
};

// ---------- Helper: compute mean of non-zero depth in ROI, convert to meters ----------
static float get_depth_value_from_bounding_box(const cv::Mat& depth_image, const BoundingBox& bb) {
    if (depth_image.empty() || depth_image.type() != CV_16UC1) return 0.f;

    auto [x1, x2] = bb.get_x_value();
    auto [y1, y2] = bb.get_y_value();

    // Clamp ROI to image bounds
    int x1c = std::max(0, std::min(x1, depth_image.cols));
    int x2c = std::max(0, std::min(x2, depth_image.cols));
    int y1c = std::max(0, std::min(y1, depth_image.rows));
    int y2c = std::max(0, std::min(y2, depth_image.rows));

    if (x2c <= x1c || y2c <= y1c) return 0.f;

    cv::Mat roi = depth_image(cv::Rect(x1c, y1c, x2c - x1c, y2c - y1c));

    // Mask out zeros (equivalent to numpy nonzero)
    cv::Mat nzMask = (roi != 0);
    if (cv::countNonZero(nzMask) == 0) return 0.f;

    // Compute mean of non-zero values (still in uint16 millimeters)
    cv::Scalar mean_mm = cv::mean(roi, nzMask);

    // Convert to meters as float32 (divide by 1000)
    float depth_m = static_cast<float>(mean_mm[0]) / 1000.0f;
    return depth_m;
}

// ---------- Set all boxes' depth fields ----------
void set_all_bounding_box_depth_values(const cv::Mat& depth_image, std::vector<BoundingBox>& boxes) {
    if (boxes.empty()) return;
    for (auto& b : boxes) {
        float d = get_depth_value_from_bounding_box(depth_image, b);
        b.set_depth(d);
    }
}

// ---------- Get intrinsics (rs2_intrinsics) ----------
rs2_intrinsics get_intrinsics() {
    // Note: works before or after start(); using resolve like your Python.
    rs2::pipeline_wrapper wrapper(g_pipeline);
    rs2::pipeline_profile prof = g_config.resolve(wrapper);
    rs2::stream_profile sp = prof.get_stream(RS2_STREAM_DEPTH);
    rs2::video_stream_profile vsp = sp.as<rs2::video_stream_profile>();
    return vsp.get_intrinsics();  // struct rs2_intrinsics { width,height,ppx,ppy,fx,fy,model,coeffs[5] }
}

// ---------- Minimal demo (optional) ----------
#ifdef DEMO_MAIN
int main() {
    initialize_real_sense();

    for (;;) {
        auto [color, depth] = get_color_depth_image();
        if (color.empty() || depth.empty()) break;

        // Example: one box centered region
        BoundingBox bb;
        bb.x1 = color.cols/2 - 40; bb.x2 = color.cols/2 + 40;
        bb.y1 = color.rows/2 - 40; bb.y2 = color.rows/2 + 40;

        std::vector<BoundingBox> boxes{bb};
        set_all_bounding_box_depth_values(depth, boxes);

        // Visualize ROI
        cv::rectangle(color, cv::Rect(bb.x1, bb.y1, bb.x2-bb.x1, bb.y2-bb.y1), {0,255,0}, 2);
        cv::putText(color, ("d=" + std::to_string(boxes[0].depth_m) + " m"),
                    {10,30}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0,255,0}, 2);

        cv::imshow("color", color);
        if ((cv::waitKey(1) & 0xFF) == 27) break;
    }
    g_pipeline.stop();
    return 0;
}
#endif

