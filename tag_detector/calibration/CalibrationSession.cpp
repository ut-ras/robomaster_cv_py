// calibrate_charuco.cpp
// Build:
//   g++ -std=c++17 calibrate_charuco.cpp -o calibrate `pkg-config --cflags --libs opencv4`
// Run from a folder that contains "manual_images/":
//   ./calibrate

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>
#include <ctime>

class CalibrationSession {
public:
    CalibrationSession() {
        dict_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_1000);
        detector_params_ = cv::aruco::DetectorParameters(); // defaults
        board_ = cv::aruco::CharucoBoard::create(12, 9, 0.019f, 0.0145f, dict_);
    }

    void process_frame(cv::Mat& image, bool save) {
        if (image.empty()) return;
        if (imsize_.width == 0 || imsize_.height == 0) imsize_ = image.size();

        std::vector<std::vector<cv::Point2f>> corners, rejected;
        std::vector<int> ids;
        cv::aruco::detectMarkers(image, dict_, corners, ids, detector_params_, rejected);
        if (corners.empty()) return;

        cv::aruco::drawDetectedMarkers(image, corners, ids);

        cv::Mat charucoCorners, charucoIds;
        bool ok = cv::aruco::interpolateCornersCharuco(corners, ids, image, board_, charucoCorners, charucoIds);
        if (ok && !charucoCorners.empty() && !charucoIds.empty()) {
            cv::aruco::drawDetectedCornersCharuco(image, charucoCorners, charucoIds);
            if (save) {
                all_charuco_corners_.push_back(charucoCorners);
                all_charuco_ids_.push_back(charucoIds);
                std::cout << "Saved calibration frame\n";
            }
        }
    }

    void finish(const std::string& calibrationFilename) {
        if (all_charuco_corners_.empty()) {
            std::cout << "ERROR: No calibration data\n";
            return;
        }

        try {
            if (std::filesystem::exists(calibrationFilename)) {
                std::filesystem::remove(calibrationFilename);
            }
        } catch (...) {}

        cv::Mat K, D;
        std::vector<cv::Mat> rvecs, tvecs;
        double reprojErr = cv::aruco::calibrateCameraCharuco(
            all_charuco_corners_, all_charuco_ids_, board_, imsize_, K, D, rvecs, tvecs);

        if (K.empty() || D.empty()) {
            std::cout << "ERROR: Calibration failed\n";
            return;
        }

        // timestamp
        std::time_t now = std::time(nullptr);
        char buf[64]{0};
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));

        cv::FileStorage fs(calibrationFilename, cv::FileStorage::WRITE);
        fs.write("calibration_date", std::string(buf));
        // Match your Python tuple: (height, width)
        fs.write("camera_resolution", cv::Vec2i(imsize_.height, imsize_.width));
        fs.write("camera_matrix", K);
        fs.write("distortion_coefficients", D);
        fs.write("reprojection_error", reprojErr);
        fs.release();

        std::cout << "Calibration finished (reproj err = " << reprojErr << ")\n";
    }

private:
    std::vector<cv::Mat> all_charuco_corners_;
    std::vector<cv::Mat> all_charuco_ids_;
    cv::Size imsize_{0,0};
    cv::Ptr<cv::aruco::Dictionary> dict_;
    cv::aruco::DetectorParameters detector_params_;
    cv::Ptr<cv::aruco::CharucoBoard> board_;
};

int main() {
    const std::string FOLDER_NAME = "manual_images";
    const std::string CALIB_FILE  = "camera_calibration.yaml";

    if (!std::filesystem::exists(FOLDER_NAME) || !std::filesystem::is_directory(FOLDER_NAME)) {
        std::cerr << "Folder \"" << FOLDER_NAME << "\" not found.\n";
        return 1;
    }

    CalibrationSession session;

    for (const auto& entry : std::filesystem::directory_iterator(FOLDER_NAME)) {
        if (!entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        if (!name.empty() && name[0] == '.') continue;

        std::cout << "Calibrating with \"" << name << "\"\n";
        cv::Mat image = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "  Could not read " << entry.path() << "\n";
            continue;
        }
        session.process_frame(image, true);

        // Optional preview
        // cv::imshow("calib", image);
        // cv::waitKey(1);
    }
    // cv::destroyAllWindows();

    session.finish(CALIB_FILE);
    std::cout << "Finished calibration\n";
    return 0;
}

