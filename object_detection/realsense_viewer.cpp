// realsense_yolov5.cpp
// Build (example):
//   g++ -std=c++17 realsense_yolov5.cpp -o rs_yolo \
//       `pkg-config --cflags --libs opencv4` -lrealsense2 \
//       -I$TORCH/include -I$TORCH/include/torch/csrc/api/include \
//       -L$TORCH/lib -ltorch -ltorch_cpu -lc10 \
//       -Wl,-rpath,$TORCH/lib
//
// Replace $TORCH with your LibTorch folder (e.g., /opt/libtorch).
// Run:
//   ./rs_yolo last.torchscript.pt

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <memory>

static rs2::pipeline g_pipeline;
static rs2::config   g_config;

static void initialize_realsense() {
    // Discover device to confirm RGB sensor exists
    rs2::pipeline_wrapper wrapper(g_pipeline);
    rs2::pipeline_profile pre_profile = g_config.resolve(wrapper);
    rs2::device dev = pre_profile.get_device();

    bool found_rgb = false;
    for (auto&& s : dev.query_sensors()) {
        if (s.supports(RS2_CAMERA_INFO_NAME) &&
            std::string(s.get_info(RS2_CAMERA_INFO_NAME)) == "RGB Camera") {
            found_rgb = true; break;
        }
    }
    if (!found_rgb) {
        std::cerr << "The demo requires Depth camera with Color sensor\n";
        std::exit(1);
    }

    // Depth 640x480@30 (unused here but harmless)
    g_config.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    // Color: pick 640x480 by default
    g_config.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);

    g_pipeline.start(g_config);
}

static torch::jit::script::Module initialize_yolo(const std::string& ts_path, torch::Device device) {
    torch::jit::script::Module m = torch::jit::load(ts_path, device);
    m.eval();
    return m;
}

// Preprocess: BGR cv::Mat -> RGB float tensor [1,3,H,W] in [0,1]
static torch::Tensor preprocess(const cv::Mat& bgr, int inp_w, int inp_h, torch::Device device) {
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(inp_w, inp_h), 0, 0, cv::INTER_LINEAR);
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    torch::Tensor t = torch::from_blob(rgb.data, {inp_h, inp_w, 3}, torch::kUInt8).to(device);
    t = t.permute({2,0,1}).contiguous().toType(torch::kFloat);
    t = t.div_(255.0f);
    return t.unsqueeze(0); // [1,3,H,W]
}

// Best-effort renderer for YOLOv5-style outputs shaped [N,6]: x1 y1 x2 y2 conf cls
static void render_if_yolo_6cols(cv::Mat& img,
                                 const torch::Tensor& det_cpu,
                                 int inp_w, int inp_h,
                                 float conf_thres = 0.25f) {
    if (det_cpu.dim() != 2 || det_cpu.size(1) < 6) return;
    auto d = det_cpu.to(torch::kCPU).contiguous();

    const int rows = (int)d.size(0);
    for (int i = 0; i < rows; ++i) {
        float x1 = d[i][0].item<float>();
        float y1 = d[i][1].item<float>();
        float x2 = d[i][2].item<float>();
        float y2 = d[i][3].item<float>();
        float conf = d[i][4].item<float>();
        float clsf = d[i][5].item<float>();

        if (conf < conf_thres) continue;
        if (x1 < 0) x1 = 0; if (y1 < 0) y1 = 0;

        int X1 = int(std::lround(x1 / float(inp_w) * img.cols));
        int Y1 = int(std::lround(y1 / float(inp_h) * img.rows));
        int X2 = int(std::lround(x2 / float(inp_w) * img.cols));
        int Y2 = int(std::lround(y2 / float(inp_h) * img.rows));

        cv::Scalar color(0,255,0);
        if (std::fabs(clsf - 0.f) < 0.5f) color = cv::Scalar(50,50,255);      // class 0
        else if (std::fabs(clsf - 1.f) < 0.5f) color = cv::Scalar(255,50,50); // class 1

        cv::rectangle(img, cv::Rect(cv::Point(X1,Y1), cv::Point(X2,Y2)), color, 2);
        char label[64];
        std::snprintf(label, sizeof(label), "c=%.0f conf=%.2f", clsf, conf);
        cv::putText(img, label, {X1, std::max(Y1-5, 0)}, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " last.torchscript.pt [inference_size]\n";
        return 1;
    }
    const std::string model_ts = argv[1];
    const int infer_size = (argc >= 3) ? std::atoi(argv[2]) : 480; // match your Python's size=480

    try {
        initialize_realsense();

        torch::Device device(torch::kCPU);
        if (torch::cuda::is_available()) device = torch::kCUDA;

        auto model = initialize_yolo(model_ts, device);

        cv::namedWindow("RealSense", cv::WINDOW_AUTOSIZE);

        while (true) {
            // Grab color frame
            rs2::frameset fs = g_pipeline.wait_for_frames();
            rs2::video_frame color = fs.get_color_frame();
            if (!color) continue;

            cv::Mat frame(cv::Size(color.get_width(), color.get_height()),
                          CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);

            // Inference
            torch::NoGradGuard _ng;
            auto input = preprocess(frame, infer_size, infer_size, device);

            // Many TorchScript YOLOv5 exports return either a list or a single tensor
            // Try to handle a few common cases:
            torch::IValue out_iv = model.forward({input});
            torch::Tensor det;

            if (out_iv.isTensor()) {
                det = out_iv.toTensor();                 // e.g., [N,6] or [1,N,6]
            } else if (out_iv.isTuple()) {
                auto tup = out_iv.toTuple();
                // often first element is the detections tensor
                for (auto& el : tup->elements()) {
                    if (el.isTensor()) { det = el.toTensor(); break; }
                }
            } else if (out_iv.isList()) {
                auto lst = out_iv.toList();
                for (size_t i = 0; i < lst.size(); ++i) {
                    if (lst.get(i).isTensor()) { det = lst.get(i).toTensor(); break; }
                }
            }

            // Print a quick summary
            std::cout << "Output dtype=" << (det.defined() ? det.dtype() : torch::kFloat)
                      << " shape=" << (det.defined() ? det.sizes() : torch::IntArrayRef{}) << "\n";

            // Normalize to [N,6] on CPU if it looks like detections
            if (det.defined()) {
                auto d = det.squeeze().to(torch::kCPU); // drop batch if present
                if (d.dim() == 2 && d.size(1) >= 6) {
                    render_if_yolo_6cols(frame, d, infer_size, infer_size, /*conf_thres*/0.25f);
                }
            }

            // Show
            cv::imshow("RealSense", frame);
            int k = cv::waitKey(1);
            if ((k & 0xFF) == 27) break; // ESC to quit
        }

        g_pipeline.stop();

    } catch (const rs2::error& e) {
        std::cerr << "RealSense error: " << e.what()
                  << " fn=" << e.get_failed_function()
                  << " args=" << e.get_failed_args() << "\n";
        return 2;
    } catch (const c10::Error& e) {
        std::cerr << "LibTorch error: " << e.msg() << "\n";
        return 3;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 4;
    }
    return 0;
}

