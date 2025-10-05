// run_inference_images_tidl.cpp
// Build (example):
//   g++ -std=c++17 run_inference_images_tidl.cpp -o run_infer \
//       `pkg-config --cflags --libs opencv4` \
//       -lonnxruntime
//
// If using TI's TIDL EP, link the EP library and include its factory header per TI docs.

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <filesystem>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <cassert>

namespace fs = std::filesystem;

static constexpr float CONFIDENCE_THRESHOLD = 0.3f;

// Draw boxes like the Python version
static void render_boxes(cv::Mat& image, int inference_w, int inference_h,
                         const float* out, int n0, int n1, int n2) {
    // Expect shape [1, count, 6] (n0==1, n1==count, n2==6)
    assert(n0 == 1);
    assert(n2 >= 6);
    for (int i = 0; i < n1; ++i) {
        const float x1 = out[(0*n1 + i)*n2 + 0];
        const float y1 = out[(0*n1 + i)*n2 + 1];
        const float x2 = out[(0*n1 + i)*n2 + 2];
        const float y2 = out[(0*n1 + i)*n2 + 3];
        const float conf = out[(0*n1 + i)*n2 + 4];
        const float clsf = out[(0*n1 + i)*n2 + 5];

        if (conf <= CONFIDENCE_THRESHOLD) continue;

        // scale back to original frame size
        int X1 = int(std::round(x1 / float(inference_w) * image.cols));
        int Y1 = int(std::round(y1 / float(inference_h) * image.rows));
        int X2 = int(std::round(x2 / float(inference_w) * image.cols));
        int Y2 = int(std::round(y2 / float(inference_h) * image.rows));

        // class colors (R,G,B) -> convert to BGR for OpenCV
        cv::Scalar color_bgr(0, 255, 0);
        if (std::fabs(clsf - 0.f) < 0.5f) color_bgr = cv::Scalar(50, 50, 255);      // (255,50,50) RGB -> BGR
        else if (std::fabs(clsf - 1.f) < 0.5f) color_bgr = cv::Scalar(255, 50, 50); // (50,50,255)  RGB -> BGR

        cv::rectangle(image, cv::Rect(cv::Point(X1, Y1), cv::Point(X2, Y2)), color_bgr, 3);
    }
}

// Convert OpenCV BGR frame to NCHW float32 RGB normalized [0,1]
static std::vector<float> preprocess_bgr_to_nchw_rgb01(const cv::Mat& bgr, int W, int H) {
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(W, H), 0, 0, cv::INTER_LINEAR); // HxW, BGR

    std::vector<float> tensor(1 * 3 * H * W);
    // Channels: R,G,B; Source is B,G,R
    for (int y = 0; y < H; ++y) {
        const uchar* row = resized.ptr<uchar>(y);
        for (int x = 0; x < W; ++x) {
            const uchar b = row[3*x + 0];
            const uchar g = row[3*x + 1];
            const uchar r = row[3*x + 2];

            const float rf = r / 255.0f;
            const float gf = g / 255.0f;
            const float bf = b / 255.0f;

            // NCHW index
            tensor[0 * (3*H*W) + 0 * (H*W) + y * W + x] = rf; // R
            tensor[0 * (3*H*W) + 1 * (H*W) + y * W + x] = gf; // G
            tensor[0 * (3*H*W) + 2 * (H*W) + y * W + x] = bf; // B
        }
    }
    return tensor;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.onnx> <tidl_artifacts_dir> <sample_images_dir>\n";
        return 1;
    }
    const std::string model_path   = argv[1];
    const std::string artifacts_dir= argv[2];
    const std::string images_dir   = argv[3];

    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "tidl_run");
        Ort::SessionOptions so;

        // OPTIONAL: set intra/interop threads, memory arenas, etc.
        so.SetIntraOpNumThreads(1);
        so.SetInterOpNumThreads(1);
        so.DisableMemPattern();
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // ---- Register/append TIDL EP here (if available) ----
        // TI's provider typically exposes a factory function to append provider with options:
        // Example (pseudo; check TI docs / headers):
        //   Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tidl(
        //       so, /*options map*/ {{"platform","J7"},{"version","8.2"},{"artifacts_folder", artifacts_dir}}
        //   ));
        //
        // If TIDL EP is not available, we simply fall back to CPU:
        // (No action needed; CPU EP is added by default in many builds. If not, call:)
        // Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CPU(so, 0));

        // Create session
        Ort::Session session(env, model_path.c_str(), so);

        // Input info
        Ort::AllocatorWithDefaultOptions alloc;
        Ort::ConstIoBinding binding(session);
        size_t num_inputs = session.GetInputCount();
        if (num_inputs != 1) {
            std::cerr << "Expected exactly 1 input, got " << num_inputs << "\n";
        }
        Ort::TypeInfo in_typeinfo = session.GetInputTypeInfo(0);
        auto in_tensor_info = in_typeinfo.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType in_type = in_tensor_info.GetElementType();
        std::vector<int64_t> in_shape = in_tensor_info.GetShape();

        // Fetch input name
        char* in_name_c = session.GetInputName(0, alloc);
        std::string input_name = in_name_c ? in_name_c : "input";
        if (in_name_c) alloc.Free(in_name_c);

        // Expect shape [N,C,H,W] with N==1 (or symbolic)
        if (in_shape.size() != 4) {
            std::cerr << "Unexpected input rank: " << in_shape.size() << "\n";
            return 2;
        }
        int64_t N = in_shape[0];
        int64_t C = in_shape[1];
        int64_t H = in_shape[2];
        int64_t W = in_shape[3];

        if (N != 1 && N != -1) {
            std::cerr << "Batch dimension must be 1 (or symbolic -1). Got " << N << "\n";
        }
        if (C != 3) {
            std::cerr << "Expected 3 channels (RGB). Got " << C << "\n";
        }
        std::cout << "Input shape: [" << N << "," << C << "," << H << "," << W << "]\n";
        std::cout << "Input name: " << input_name << "\n";

        // Load images
        std::vector<std::string> image_paths;
        for (auto& p : fs::directory_iterator(images_dir)) {
            if (p.is_regular_file()) image_paths.push_back(p.path().string());
        }
        if (image_paths.empty()) {
            std::cerr << "No images found in " << images_dir << "\n";
            return 3;
        }

        // Preprocess all images
        std::vector<std::vector<float>> preprocessed;
        preprocessed.reserve(image_paths.size());
        std::vector<cv::Mat> originals;
        originals.reserve(image_paths.size());

        for (const auto& path : image_paths) {
            cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
            if (bgr.empty()) {
                std::cerr << "Failed to read: " << path << "\n";
                continue;
            }
            originals.push_back(bgr);
            preprocessed.push_back(preprocess_bgr_to_nchw_rgb01(bgr, int(W), int(H)));
        }

        // Prepare output dir
        fs::create_directories("sample_detections");

        // Time multiple reps (like Python)
        const int NUM_TIMING_REPS = 200;
        auto t0 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < NUM_TIMING_REPS; ++rep) {
            for (size_t i = 0; i < preprocessed.size(); ++i) {
                // Create input tensor (shape [1,3,H,W])
                std::array<int64_t,4> shape{1, 3, H, W};
                Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                    alloc, preprocessed[i].data(), preprocessed[i].size(), shape.data(), shape.size()
                );

                // Run
                auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                                  &input_name, &input_tensor, 1,
                                                  nullptr, 0); // fetch all outputs
                (void)output_tensors; // timing loop; ignore result
            }
        }
        auto t1 = std::chrono::steady_clock::now();
        const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double total_execs = double(NUM_TIMING_REPS) * double(preprocessed.size());
        std::cout << "Time per inference (ms): " << (total_ms / total_execs) << "\n";

        // Do one real pass and save detections (assumes single output head)
        for (size_t i = 0; i < preprocessed.size(); ++i) {
            std::array<int64_t,4> shape{1, 3, H, W};
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                alloc, preprocessed[i].data(), preprocessed[i].size(), shape.data(), shape.size()
            );
            auto outputs = session.Run(Ort::RunOptions{nullptr},
                                       &input_name, &input_tensor, 1,
                                       nullptr, 0);
            if (outputs.empty() || !outputs[0].IsTensor()) {
                std::cerr << "Unexpected outputs for image #" << i << "\n";
                continue;
            }
            auto& out0 = outputs[0];
            auto out_info = out0.GetTensorTypeAndShapeInfo();
            auto out_shape = out_info.GetShape();   // expect [1, count, 6] or [1, count, _, 6]
            const float* out_data = out0.GetTensorData<float>();

            // Normalize shape to [1,count,6]
            int n0 = (out_shape.size() >= 1) ? int(out_shape[0]) : 1;
            int n1 = (out_shape.size() >= 2) ? int(out_shape[1]) : 0;
            int n2 = (out_shape.size() >= 3) ? int(out_shape.back()) : 0;
            if (out_shape.size() == 4) {
                // e.g., [1, count, 1, 6] → fold middle dim
                n2 = int(out_shape[3]);
                // We will treat as [1,count,6] by ignoring the 3rd dim if it's 1
                if (out_shape[2] != 1) {
                    std::cerr << "Unexpected 4D output shape; expected third dim == 1.\n";
                }
            }

            cv::Mat vis = originals[i].clone();
            render_boxes(vis, int(W), int(H), out_data, n0, n1, n2);

            const std::string stem = fs::path(image_paths[i]).stem().string();
            const std::string outpath = (fs::path("sample_detections") / (stem + ".png")).string();
            cv::imwrite(outpath, vis);
        }

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << "\n";
        return 10;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 11;
    }
    return 0;
}

