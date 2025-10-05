// calibrate.cpp
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

int main() {
    // Checkerboard inner-corner dimensions (cols, rows)
    // Matches Python's CHECKERBOARD = (6, 9)
    const cv::Size CHECKERBOARD(6, 9);

    // Termination criteria (EPS + MAX_ITER, 30 iters, epsilon=1e-3)
    const cv::TermCriteria criteria(
        cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 1e-3
    );

    // Collect file paths (cv::glob handles wildcards and spaces in the folder name)
    std::vector<cv::String> imagePaths;
    cv::glob("realsense calibration/*.jpg", imagePaths, false);

    if (imagePaths.empty()) {
        std::cerr << "No images found at: realsense calibration/*.jpg\n";
        return 1;
    }

    // Storage for all images' points
    std::vector<std::vector<cv::Point3f>> objectPoints;  // 3D points in world space
    std::vector<std::vector<cv::Point2f>> imagePoints;   // 2D points in image plane

    // Prepare one pattern of 3D object points (z = 0)
    std::vector<cv::Point3f> objp;
    objp.reserve(CHECKERBOARD.width * CHECKERBOARD.height);
    for (int y = 0; y < CHECKERBOARD.height; ++y) {
        for (int x = 0; x < CHECKERBOARD.width; ++x) {
            objp.emplace_back(static_cast<float>(x), static_cast<float>(y), 0.0f);
        }
    }

    cv::Size imageSize;
    cv::Mat lastImage, lastGray;

    for (const auto& path : imagePaths) {
        cv::Mat image = cv::imread(path);
        if (image.empty()) {
            std::cerr << "Failed to read: " << path << "\n";
            continue;
        }
        imageSize = image.size();  // keep updated; used for calibration

        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        // Find chessboard corners
        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(
            gray, CHECKERBOARD, corners,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE
        );

        if (found) {
            // Refine corner positions to sub-pixel accuracy
            cv::cornerSubPix(
                gray, corners, cv::Size(11, 11), cv::Size(-1, -1), criteria
            );

            // Save corresponding points
            objectPoints.push_back(objp);
            imagePoints.push_back(corners);

            // Draw and show
            cv::drawChessboardCorners(image, CHECKERBOARD, corners, found);
        } else {
            std::cerr << "Checkerboard not found in: " << path << "\n";
        }

        cv::imshow("img", image);
        cv::waitKey(0); // press any key to advance
        lastImage = image;
        lastGray = gray;
    }

    cv::destroyAllWindows();

    if (objectPoints.empty() || imagePoints.empty()) {
        std::cerr << "No valid detections. Calibration aborted.\n";
        return 1;
    }

    // Run calibration
    cv::Mat cameraMatrix, distCoeffs;
    std::vector<cv::Mat> rvecs, tvecs;

    double reprojErr = cv::calibrateCamera(
        objectPoints, imagePoints, imageSize,
        cameraMatrix, distCoeffs, rvecs, tvecs
    );

    // Output results
    std::cout << "Reprojection error:\n" << reprojErr << "\n\n";
    std::cout << "Camera matrix:\n" << cameraMatrix << "\n\n";
    std::cout << "Distortion coefficients:\n" << distCoeffs << "\n\n";

    std::cout << "Rotation vectors (Rodrigues):\n";
    for (size_t i = 0; i < rvecs.size(); ++i) {
        std::cout << "rvec[" << i << "]:\n" << rvecs[i] << "\n";
    }
    std::cout << "\nTranslation vectors:\n";
    for (size_t i = 0; i < tvecs.size(); ++i) {
        std::cout << "tvec[" << i << "]:\n" << tvecs[i] << "\n";
    }

    return 0;
}

