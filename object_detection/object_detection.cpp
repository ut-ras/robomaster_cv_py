// object_detector.cpp
// Build example:
// g++ -std=c++17 object_detector.cpp -o obj_det `pkg-config --cflags --libs opencv4` -lonnxruntime
//
// If you have TI’s TIDL EP, link it and call its Append factory where noted below.

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <cmath>
#include <cassert>

// ---- Your BoundingBox class header here ----
// #include "BoundingBox.hpp"
class BoundingBox {
public:
    void set_x_value(int x1, int x2) { x1_ = x1; x2_ = x2; }
    void set_y_value(int y1, int y2) { y1_ = y1; y2_ = y2; }
    void calculate_height() { height_ = y2_ - y1_; }
    void calculate_width()  { width_  = x2_ - x1_; }
private:
    int x1_{0}, x2_{0}, y1_{0}, y2_{0}, height_{0}, width_{0};
};

class object_detector {
public:
    object_detector(const std::string& model_path,
                    const std::string& artifacts_dir)
        : model_path_(model_path), artifacts_dir_(artifacts_dir) {}

    void initialize_object_detections() {
        // Session options
        Ort::SessionOptions so;
        so.SetIntraOpNumThreads(1);
        so.SetInterOpNumThreads(1);
        so.DisableMemPattern();
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Print available EPs
        {
            size_t count = Ort::GetAvailableProviders().size();
            std::cerr << "Available EPs: ";
            const auto eps = Ort::GetAvailableProviders();
            for (size_t i = 0; i < eps.size(); ++i) {
                std::cerr << eps[i] << (i + 1 < eps.size() ? ", " : "");
            }
            std::cerr << "\n";
        }

        // ---- Append TIDL EP here (if your ORT build includes it). Example (pseudo):
        // Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tidl(
        //     so,
        //     { {"platform","J7"}, {"version","8.2"}, {"artifacts_folder", artifacts_dir_} }
        // ));
        // If you skip this, it will run on CPU EP.

        // Create session
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "objdet");
        session_ = std::make_unique<Ort::Session>(*env_, model_path_.c_str(), so);

        // Read input info
        Ort::AllocatorWithDefaultOptions alloc;
        Ort::TypeInfo ti = session_->GetInputTypeInfo(0);
        auto tt = ti.GetTensorTypeAndShapeInfo();
        input_name_.reset(session_->GetInputName(0, alloc)); // freed by custom deleter below

        auto shape = tt.GetShape(); // expect [N,C,H,W]
        if (shape.size() != 4) throw std::runtime_error("Unexpected input rank");
        int64_t N = shape[0], C = shape[1];
        H_ = (int)shape[2];
        W_ = (int)shape[3];

        if (!(N == 1 || N == -1)) std::cerr << "Warning: N != 1 or -1\n";
        if (C != 3) throw std::runtime_error("Expected C==3");

        std::cout << "Input shape: [" << N << "," << C << "," << H_ << "," << W_ << "]\n";
        std::cout << "Input name: " << input_name_.get() << "\n";
    }

    // Runs detection: fills boundingbox_list and draws boxes on the passed image.
    void run_object_detections(cv::Mat& image, std::vector<BoundingBox>& boundingbox_list) {
        // BGR -> RGB, resize to WxH, NCHW, /255
        std::vector<float> tensor = preprocess_bgr_to_nchw_rgb01(image, W_, H_);

        std::array<int64_t, 4> ishape{1, 3, H_, W_};
        Ort::AllocatorWithDefaultOptions alloc;
        Ort::Value input = Ort::Value::CreateTensor<float>(
            alloc, tensor.data(), tensor.size(), ishape.data(), ishape.size()
        );

        const char* in_names[] = { input_name_.get() };
        auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                     in_names, &input, 1,
                                     nullptr, 0); // get all outputs

        if (outputs.empty() || !outputs[0].IsTensor()) {
            std::cerr << "Unexpected outputs\n";
            return;
        }

        auto& out0 = outputs[0];
        auto out_info  = out0.GetTensorTypeAndShapeInfo();
        auto out_shape = out_info.GetShape();  // often [1, count, 6] or [1,count,1,6]
        const float* out_data = out0.GetTensorData<float>();

        int n0 = (out_shape.size() >= 1) ? (int)out_shape[0] : 1;
        int n1 = (out_shape.size() >= 2) ? (int)out_shape[1] : 0;
        int n2 = (out_shape.size() >= 3) ? (int)out_shape.back() : 0;
        if (out_shape.size() == 4) {
            // e.g., [1,count,1,6] -> treat as [1,count,6]
            n2 = (int)out_shape[3];
        }
        // slice pointer to [1,count,6] semantics
        render_boxes(image, out_data, n0, n1, n2, boundingbox_list);
    }

private:
    // owns the input name C-string
    struct FreeOrtChar {
        void operator()(char* p) const noexcept { Ort::AllocatorWithDefaultOptions{}.Free(p); }
    };

    static std::vector<float> preprocess_bgr_to_nchw_rgb01(const cv::Mat& bgr, int W, int H) {
        cv::Mat resized;
        cv::resize(bgr, resized, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);
        std::vector<float> t(1 * 3 * H * W);

        for (int y = 0; y < H; ++y) {
            const uchar* row = resized.ptr<uchar>(y);
            for (int x = 0; x < W; ++x) {
                const uchar B = row[3*x + 0];
                const uchar G = row[3*x + 1];
                const uchar R = row[3*x + 2];
                const float rf = R / 255.f, gf = G / 255.f, bf = B / 255.f;

                // NCHW
                t[0 * (3*H*W) + 0 * (H*W) + y * W + x] = rf; // R
                t[0 * (3*H*W) + 1 * (H*W) + y * W + x] = gf; // G
                t[0 * (3*H*W) + 2 * (H*W) + y * W + x] = bf; // B
            }
        }
        return t;
    }

    void render_boxes(cv::Mat& image,
                      const float* out, int n0, int n1, int n2,
                      std::vector<BoundingBox>& bbox_list) const {
        constexpr float CONFIDENCE_THRESHOLD = 0.2f;
        assert(n0 == 1);
        if (n2 < 6) return;

        for (int i = 0; i < n1; ++i) {
            float x1 = out[(0*n1 + i)*n2 + 0];
            float y1 = out[(0*n1 + i)*n2 + 1];
            float x2 = out[(0*n1 + i)*n2 + 2];
            float y2 = out[(0*n1 + i)*n2 + 3];
            float conf = out[(0*n1 + i)*n2 + 4];
            float clsf = out[(0*n1 + i)*n2 + 5];

            if (conf < CONFIDENCE_THRESHOLD) continue;

            // clamp negatives like Python
            if (x1 < 0.f) x1 = 0.f;
            if (y1 < 0.f) y1 = 0.f;

            std::cout << "x1 " << x1 << "\n"
                      << "y1 " << y1 << "\n"
                      << "x2 " << x2 << "\n"
                      << "y2 " << y2 << "\n"
                      << "confidence " << conf << "\n"
                      << "class_idx_float " << clsf << "\n";

            // scale to original image
            int X1 = (int)std::lround(x1 / float(W_) * image.cols);
            int Y1 = (int)std::lround(y1 / float(H_) * image.rows);
            int X2 = (int)std::lround(x2 / float(W_) * image.cols);
            int Y2 = (int)std::lround(y2 / float(H_) * image.rows);

            // Colors given as (R,G,B) in Python; convert to BGR for OpenCV
            cv::Scalar color_bgr(0, 255, 0);
            if (std::fabs(clsf - 0.f) < 0.5f) color_bgr = cv::Scalar(50, 50, 255);      // (255,50,50)->BGR
            else if (std::fabs(clsf - 1.f) < 0.5f) color_bgr = cv::Scalar(255, 50, 50); // (50,50,255)->BGR

            cv::rectangle(image, cv::Rect(cv::Point(X1, Y1), cv::Point(X2, Y2)), color_bgr, 3);

            BoundingBox b;
            b.set_x_value(X1, X2);
            b.set_y_value(Y1, Y2);
            b.calculate_height();
            b.calculate_width();
            bbox_list.push_back(b);
        }
    }

private:
    std::string model_path_;
    std::string artifacts_dir_;
    std::unique_ptr<Ort::Env>    env_;
    std::unique_ptr<Ort::Session> session_;

    int H_{0}, W_{0};
    std::unique_ptr<char, FreeOrtChar> input_name_;

    // Accessors equivalent to Python __height__/__width__
    int H_() const { return H_; }
    int W_() const { return W_; }
};

//
// Usage example:
//
// int main() {
//     object_detector det("/home/debian/robomaster_CV/object_detection/last_with_shapes.onnx",
//                         "/home/debian/robomaster_CV/object_detection/tidl_output");
//     det.initialize_object_detections();
//
//     cv::Mat img = cv::imread("some.jpg");
//     std::vector<BoundingBox> boxes;
//     det.run_object_detections(img, boxes);
//     cv::imwrite("out.png", img);
// }

