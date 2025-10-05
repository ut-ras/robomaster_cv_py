// tests_realsense.cpp
// Build (example):
//   g++ -std=c++17 tests_realsense.cpp -o tests \
//       -lrealsense2 -lgtest -lpthread
//
// Or use CMake (snippet below).

#include <gtest/gtest.h>
#include <librealsense2/rs.hpp>
#include <iostream>

// Globals similar to your Python module-level objects
static rs2::pipeline g_pipeline;
static rs2::config   g_config;
static rs2::align    g_align(RS2_STREAM_COLOR);

static void initialize_real_sense() {
    // Configure streams to 1280x720 (as in your test expectations)
    // Depth
    g_config.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16, 30);
    // Color
    g_config.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8, 30);

    // Resolve before start to inspect device and confirm RGB presence
    rs2::pipeline_wrapper wrapper(g_pipeline);
    rs2::pipeline_profile pre_profile = g_config.resolve(wrapper);
    rs2::device dev = pre_profile.get_device();

    bool found_rgb = false;
    for (auto &&s : dev.query_sensors()) {
        if (s.supports(RS2_CAMERA_INFO_NAME) &&
            std::string(s.get_info(RS2_CAMERA_INFO_NAME)) == "RGB Camera") {
            found_rgb = true;
            break;
        }
    }
    ASSERT_TRUE(found_rgb) << "The demo requires Depth camera with Color sensor";

    // Start pipeline
    g_pipeline.start(g_config);
}

// Helper to grab one aligned color+depth pair
static std::pair<rs2::video_frame, rs2::depth_frame> get_aligned_frames() {
    rs2::frameset fs = g_pipeline.wait_for_frames();
    rs2::frameset aligned = g_align.process(fs);
    rs2::video_frame color = aligned.get_color_frame();
    rs2::depth_frame depth = aligned.get_depth_frame();
    return {color, depth};
}

// ---------- Tests ----------

TEST(RealSense, ConnectionHasRGB) {
    initialize_real_sense();

    // Re-check device for RGB sensor (mirrors the Python test body)
    rs2::pipeline_wrapper wrapper(g_pipeline);
    rs2::pipeline_profile pre_profile = g_config.resolve(wrapper);
    rs2::device dev = pre_profile.get_device();

    bool found_rgb = false;
    for (auto &&s : dev.query_sensors()) {
        if (s.supports(RS2_CAMERA_INFO_NAME) &&
            std::string(s.get_info(RS2_CAMERA_INFO_NAME)) == "RGB Camera") {
            found_rgb = true;
            break;
        }
    }
    EXPECT_TRUE(found_rgb);
}

TEST(RealSense, DepthFrameIs1280x720) {
    // Make sure pipeline is running (safe if already started)
    // (initialize_real_sense() called in previous test, but call defensively if needed)
    // initialize_real_sense(); // uncomment if running tests individually

    // Grab a frame and check depth resolution
    auto [color, depth] = get_aligned_frames();
    ASSERT_TRUE(color);  // not null
    ASSERT_TRUE(depth);

    EXPECT_EQ(depth.get_width(),  1280);
    EXPECT_EQ(depth.get_height(),  720);
}

// ---------- Test main ----------
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    try { g_pipeline.stop(); } catch (...) {}
    return ret;
}

