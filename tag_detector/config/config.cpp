// ConfigStore.hpp
#pragma once
#include <string>
#include <opencv2/core.hpp>

struct LocalConfig {
    std::string device_id = "";
    std::string server_ip = "";
    int         stream_port = 8000;

    bool has_calibration = false;

    // Matches numpy float64 (CV_64F). Leave empty until filled.
    cv::Mat camera_matrix;            // expected 3x3, CV_64F
    cv::Mat distortion_coefficients;  // expected Nx1 or 1xN, CV_64F

    LocalConfig() = default;
};

struct RemoteConfig {
    int   camera_id = -1;
    int   camera_resolution_width  = 0;
    int   camera_resolution_height = 0;
    int   camera_auto_exposure = 0;  // 0/1 or driver-specific
    int   camera_exposure = 0;       // units depend on camera API
    int   camera_gain = 0;
    double fiducial_size_m = 0.0;

    // If you’ll load structured layouts, consider using nlohmann::json here.
    // Keeping it empty/placeholder by default.
    // Replace with your own type if you have one.
    // Example: nlohmann::json tag_layout;
    // For now we keep an empty cv::Mat or std::string placeholder.
    std::string tag_layout; // optional placeholder

    RemoteConfig() = default;
};

struct ConfigStore {
    LocalConfig  local_config;
    RemoteConfig remote_config;
};

