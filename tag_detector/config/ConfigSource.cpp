// file_config_source.cpp
// Build (Ubuntu):
//   g++ -std=c++17 file_config_source.cpp -o cfg `pkg-config --cflags --libs opencv4`
//   # Needs nlohmann/json single-header (install: sudo apt-get install nlohmann-json3-dev)
//   # Or place "json.hpp" next to this file and change the include below accordingly.

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <string>

// ---------------- Mock ConfigStore (match your Python fields) ----------------
struct LocalConfig {
    std::string device_id;
    std::string server_ip;
    int         stream_port = 0;

    cv::Mat camera_matrix;
    cv::Mat distortion_coefficients;
    bool    has_calibration = false;
};

struct ConfigStore {
    LocalConfig local_config;
};

// ---------------- Interfaces ----------------
struct ConfigSource {
    virtual ~ConfigSource() = default;
    virtual void update(ConfigStore& config_store) = 0;
};

struct FileConfigSource : public ConfigSource {
    static inline const std::string CONFIG_FILENAME      = "config.json";
    static inline const std::string CALIBRATION_FILENAME = "calibration.json";

    void update(ConfigStore& config_store) override {
        // ---- Read config.json (device_id, server_ip, stream_port) ----
        nlohmann::json j;
        {
            std::ifstream f(CONFIG_FILENAME);
            if (!f) {
                throw std::runtime_error("Could not open " + CONFIG_FILENAME);
            }
            f >> j;
        }

        // Basic validation with graceful defaults
        if (!j.contains("device_id") || !j.contains("server_ip") || !j.contains("stream_port")) {
            throw std::runtime_error("Missing required keys in " + CONFIG_FILENAME +
                                     " (need device_id, server_ip, stream_port)");
        }

        config_store.local_config.device_id   = j.at("device_id").get<std::string>();
        config_store.local_config.server_ip   = j.at("server_ip").get<std::string>();
        config_store.local_config.stream_port = j.at("stream_port").get<int>();

        // ---- Read calibration.json with OpenCV FileStorage ----
        // (OpenCV can read YAML/XML/JSON as long as it’s in FileStorage format.)
        cv::FileStorage fs(CALIBRATION_FILENAME, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            // No calibration file – just leave has_calibration=false
            return;
        }

        cv::Mat K, D;
        fs["camera_matrix"] >> K;
        fs["distortion_coefficients"] >> D;
        fs.release();

        if (!K.empty() && !D.empty() && K.rows == 3 && K.cols == 3) {
            config_store.local_config.camera_matrix = K;
            config_store.local_config.distortion_coefficients = D;
            config_store.local_config.has_calibration = true;
        }
    }
};

// ---------------- Example usage / quick test ----------------
#ifdef DEMO_MAIN
int main() {
    try {
        ConfigStore store;
        FileConfigSource src;
        src.update(store);

        std::cout << "device_id: "   << store.local_config.device_id   << "\n";
        std::cout << "server_ip: "   << store.local_config.server_ip   << "\n";
        std::cout << "stream_port: " << store.local_config.stream_port << "\n";
        std::cout << "has_calibration: " << std::boolalpha << store.local_config.has_calibration << "\n";
        if (store.local_config.has_calibration) {
            std::cout << "camera_matrix:\n" << store.local_config.camera_matrix << "\n";
            std::cout << "distortion_coefficients:\n" << store.local_config.distortion_coefficients << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
#endif

