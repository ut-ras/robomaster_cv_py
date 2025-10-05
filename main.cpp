// main.cpp
// g++ -std=c++17 main.cpp -o app `pkg-config --cflags --libs opencv4`
// (Replace stubs in the classes with your actual logic)

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <optional>

// ---------- Types ----------
struct Intrinsics {
    // Pinhole intrinsics (replace with RealSense intrinsics if you have them)
    float fx{600.f}, fy{600.f}, cx{320.f}, cy{240.f};
    float depth_scale{1.f}; // if your depth is in meters already, keep 1.f
};

struct BoundingBox {
    // Minimal BB with a 3D point attached
    int x{0}, y{0}, w{0}, h{0};
    float depth_m{0.f};     // depth at the region (meters)
    float X{0.f}, Y{0.f}, Z{0.f}; // 3D coordinates

    float get_x_coord() const { return X; }
    float get_y_coord() const { return Y; }
    float get_z_coord() const { return Z; }
};

// ---------- depth_calculation (dp) ----------
namespace dp {
    static std::optional<Intrinsics> g_intr;

    void initialize_real_sense() {
        // TODO: real init of your camera (e.g., rs2::pipeline start)
        // For now just set intrinsics placeholder
        g_intr = Intrinsics{};
        std::cout << "[dp] RealSense initialized\n";
    }

    Intrinsics get_intrinsics() {
        if (!g_intr) g_intr = Intrinsics{};
        return *g_intr;
    }

    std::pair<cv::Mat, cv::Mat> get_color_depth_image() {
        // TODO: Replace with camera grab
        // For demo: create dummy 640x480 color + depth
        cv::Mat color(480, 640, CV_8UC3, cv::Scalar(30, 30, 30));
        cv::putText(color, "Dummy frame", {40, 80}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {255,255,255}, 2);
        cv::Mat depth(480, 640, CV_32FC1, cv::Scalar(2.0f)); // 2 meters everywhere
        return {color, depth};
    }

    void set_all_bounding_box_depth_values(const cv::Mat& depth, std::vector<BoundingBox>& boxes) {
        // Example: take median depth inside the box (here, simple center pixel)
        for (auto& b : boxes) {
            int u = std::clamp(b.x + b.w / 2, 0, depth.cols - 1);
            int v = std::clamp(b.y + b.h / 2, 0, depth.rows - 1);
            float d = depth.at<float>(v, u);
            b.depth_m = d;
        }
    }
}

// ---------- object_detection (od) ----------
namespace od {
    struct object_detector {
        void initialize_object_detections() {
            // TODO: load model / warmup
            std::cout << "[od] Detector initialized\n";
        }

        void run_object_detections(const cv::Mat& color, std::vector<BoundingBox>& out_boxes) {
            // TODO: run your detector and fill out_boxes
            out_boxes.clear();
            // Demo: emit one fixed box so the loop exercises the rest of the pipeline
            int w = color.cols, h = color.rows;
            BoundingBox bb;
            bb.x = w/2 - 40; bb.y = h/2 - 40; bb.w = 80; bb.h = 80;
            out_boxes.push_back(bb);
        }
    };
}

// ---------- pixel_to_point (ptp) ----------
namespace ptp {
    void set_point_coords(std::vector<BoundingBox>& boxes, const Intrinsics& K) {
        for (auto& b : boxes) {
            // Back-project center pixel of the box
            float u = static_cast<float>(b.x + b.w / 2);
            float v = static_cast<float>(b.y + b.h / 2);
            float Z = b.depth_m * K.depth_scale; // meters
            if (Z <= 0.f || K.fx == 0.f || K.fy == 0.f) {
                b.X = b.Y = b.Z = 0.f;
                continue;
            }
            float X = (u - K.cx) * Z / K.fx;
            float Y = (v - K.cy) * Z / K.fy;
            b.X = X; b.Y = Y; b.Z = Z;
        }
    }
}

// ---------- communication (com) ----------
namespace com {
    void initialize_communication() {
        // TODO: open socket/serial/etc.
        std::cout << "[com] Communication initialized\n";
    }

    void send_turret_data(float xPos, float yPos, float zPos,
                          float xVel, float yVel, float zVel,
                          float xAcc, float yAcc, float zAcc,
                          bool hasTarget) {
        // TODO: replace with your protocol send
        std::cout << "[com] hasTarget=" << (hasTarget ? "true" : "false")
                  << " pos=(" << xPos << "," << yPos << "," << zPos << ")"
                  << " vel=(" << xVel << "," << yVel << "," << zVel << ")"
                  << " acc=(" << xAcc << "," << yAcc << "," << zAcc << ")\n";
    }
}

// ---------- run_forever (main loop) ----------
static void run_forever() {
    dp::initialize_real_sense();
    od::object_detector detector;
    detector.initialize_object_detections();
    com::initialize_communication();

    std::vector<BoundingBox> boundingbox_list;
    Intrinsics intrinsics = dp::get_intrinsics();

    for (;;) {
        // Acquire frames
        auto [color_image, depth_image] = dp::get_color_depth_image();

        // Detect
        detector.run_object_detections(color_image, boundingbox_list);

        if (boundingbox_list.empty()) {
            com::send_turret_data(
                /*xPos*/0.f, /*yPos*/0.f, /*zPos*/0.f,
                /*xVel*/0.f, /*yVel*/0.f, /*zVel*/0.f,
                /*xAcc*/0.f, /*yAcc*/0.f, /*zAcc*/0.f,
                /*hasTarget*/false
            );
            boundingbox_list.clear();
            // continue;
        } else {
            // Depth for each box
            dp::set_all_bounding_box_depth_values(depth_image, boundingbox_list);

            // Pixel -> 3D point
            ptp::set_point_coords(boundingbox_list, intrinsics);

            // Send first target (match your Python)
            const auto& b0 = boundingbox_list[0];
            com::send_turret_data(
                b0.get_x_coord(), b0.get_y_coord(), b0.get_z_coord(),
                /*xVel*/0.f, /*yVel*/0.f, /*zVel*/0.f,
                /*xAcc*/0.f, /*yAcc*/0.f, /*zAcc*/0.f,
                /*hasTarget*/true
            );
            boundingbox_list.clear();
        }

        // Optional: show the image for debugging
        // cv::imshow("RealSense", color_image);
        // if (cv::waitKey(1) == 27) break; // ESC to exit
    }
}

int main() {
    run_forever();
    return 0;
}

