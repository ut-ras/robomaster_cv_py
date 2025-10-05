// main.cpp
// Build (Ubuntu):
//   g++ -std=c++17 main.cpp -o calibrate `pkg-config --cflags --libs opencv4`

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

#ifndef HAVE_CALIBRATION_SESSION
// ---- Stub; replace with your real class header/impl ----
class CalibrationSession {
public:
    void process_frame(const cv::Mat& image, bool some_flag) {
        // TODO: run your calibration step here
        (void)some_flag;
        if (image.empty()) {
            std::cerr << "Warning: empty image\n";
        }
    }
    void finish() {
        // TODO: finalize calibration (save params, etc.)
    }
};
#endif

static const std::string FOLDER_NAME = "manual_images";

int main() {
    CalibrationSession calibration_session;

    if (!fs::exists(FOLDER_NAME) || !fs::is_directory(FOLDER_NAME)) {
        std::cerr << "Error: folder \"" << FOLDER_NAME << "\" not found.\n";
        return 1;
    }

    // Iterate files (skip dotfiles)
    for (const auto& entry : fs::directory_iterator(FOLDER_NAME)) {
        if (!entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        if (!name.empty() && name[0] == '.') continue;

        std::cout << "Calibrating with \"" << name << "\"\n";
        cv::Mat image = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "  Could not read image: " << entry.path() << "\n";
            continue;
        }
