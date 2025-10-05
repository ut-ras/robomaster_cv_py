// tag_localization.cpp
// Build (Ubuntu):
//   g++ -std=c++17 tag_localization.cpp -o tag_loc `pkg-config --cflags --libs opencv4`
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

static constexpr float TAG_SIZE = 5.0f; // cm

// Intrinsics from cap (fx=fy=980; cx,cy at image center). Distortion = 0
static void get_camera_matrix(cv::VideoCapture& cap, cv::Mat& K, cv::Mat& dist) {
    double width  = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    double height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double f = 980.0;

    K = (cv::Mat_<float>(3,3) << (float)f, 0.f, (float)(width/2.0),
                                  0.f, (float)f, (float)(height/2.0),
                                  0.f, 0.f, 1.f);
    dist = cv::Mat::zeros(4, 1, CV_32F);
}

// Tag corners in its own frame (Z=0 plane), units in cm
static std::vector<cv::Point3f> get_object_points(float tag_size_cm) {
    float h = tag_size_cm * 0.5f;
    return {
        {-h, -h, 0.f},
        { h, -h, 0.f},
        { h,  h, 0.f},
        {-h,  h, 0.f},
    };
}

// Try to find a red convex quad; return 4 points (float) or empty
static std::vector<cv::Point2f> detect_red_target(const cv::Mat& bgr) {
    cv::Mat hsv; cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

    cv::Mat mask1, mask2;
    // Lower red range
    cv::inRange(hsv, cv::Scalar(0,120,70),   cv::Scalar(10,255,255), mask1);
    // Upper red range
    cv::inRange(hsv, cv::Scalar(170,120,70), cv::Scalar(180,255,255), mask2);

    cv::Mat mask = mask1 | mask2;
    cv::Mat red_regions; cv::bitwise_and(bgr, bgr, red_regions, mask);

    cv::Mat gray; cv::cvtColor(red_regions, gray, cv::COLOR_BGR2GRAY);
    cv::Mat edges; cv::Canny(gray, edges, 50, 150);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::sort(contours.begin(), contours.end(),
              [](const auto& a, const auto& b){ return cv::contourArea(a) > cv::contourArea(b); });

    for (const auto& c : contours) {
        double peri = cv::arcLength(c, true);
        std::vector<cv::Point> approx;
        cv::approxPolyDP(c, approx, 0.02 * peri, true);
        if (approx.size() == 4 && cv::isContourConvex(approx)) {
            if (cv::contourArea(approx) > 500.0) {
                std::vector<cv::Point2f> corners;
                corners.reserve(4);
                for (auto& p : approx) corners.emplace_back((float)p.x, (float)p.y);
                return corners; // in some order; solvePnP tolerates if consistent with object points
            }
        }
    }
    return {};
}

// Draw 3D axes with origin at 'corner0' (projected)
static void draw_axes(cv::Mat& frame,
                      const cv::Mat& rvec, const cv::Mat& tvec,
                      const cv::Mat& K, const cv::Mat& dist,
                      const cv::Point2f& corner0)
{
    std::vector<cv::Point3f> axis = {
        {0.f,0.f,0.f}, {2.f,0.f,0.f}, {0.f,2.f,0.f}, {0.f,0.f,-2.f}
    };
    std::vector<cv::Point2f> imgpts;
    cv::projectPoints(axis, rvec, tvec, K, dist, imgpts);

    cv::Point origin((int)corner0.x, (int)corner0.y);
    for (size_t i = 1; i < imgpts.size(); ++i) {
        cv::line(frame, origin, imgpts[i], cv::Scalar(0,0,255), 2);
    }
}

// Invert camera->tag pose to get tag->camera (i.e., camera in tag frame)
static void invert_pose(const cv::Mat& rvec, const cv::Mat& tvec,
                        cv::Mat& rvec_inv, cv::Mat& tvec_inv)
{
    cv::Mat R; cv::Rodrigues(rvec, R);
    cv::Mat R_inv = R.t();
    tvec_inv = -R_inv * tvec; // 3x1
    cv::Rodrigues(R_inv, rvec_inv);
}

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open webcam.\n";
        return 1;
    }

    cv::Mat K, dist;
    get_camera_matrix(cap, K, dist);
    auto obj_pts = get_object_points(TAG_SIZE);

    std::vector<cv::Point3f> camera_positions; // store as (x,y,z) in cm (tag frame)
    int update_counter = 0;

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame)) break;

        auto corners = detect_red_target(frame);
        if (!corners.empty()) {
            // Draw polygon for debug
            std::vector<std::vector<cv::Point>> poly(1);
            for (auto& p : corners) poly[0].push_back(cv::Point((int)p.x,(int)p.y));
            cv::polylines(frame, poly, true, cv::Scalar(0,255,0), 2);

            // Need 2D points order to match object points order. If needed, you can sort corners
            // (e.g., by cv::cornerSubPix + consistent winding). We'll assume approx order is fine.
            cv::Mat rvec, tvec;
            bool ok = cv::solvePnP(obj_pts, corners, K, dist, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
            if (ok) {
                // Reprojection error check
                std::vector<cv::Point2f> proj;
                cv::projectPoints(obj_pts, rvec, tvec, K, dist, proj);

                double err = 0.0;
                for (int i = 0; i < 4; ++i) {
                    cv::Point2f d = corners[i] - proj[i];
                    err += std::sqrt(d.x*d.x + d.y*d.y);
                }
                err /= 4.0;

                if (err < 5.0) { // pixels
                    cv::Mat rvec_inv, tvec_inv;
                    invert_pose(rvec, tvec, rvec_inv, tvec_inv);

                    // camera position in tag frame (in same units as obj_pts => cm)
                    float cx = (float)tvec_inv.at<double>(0,0);
                    float cy = (float)tvec_inv.at<double>(1,0);
                    float cz = (float)tvec_inv.at<double>(2,0);

                    if (!camera_positions.empty()) {
                        cv::Point3f last = camera_positions.back();
                        float dx = cx - last.x, dy = cy - last.y, dz = cz - last.z;
                        float dist_cm = std::sqrt(dx*dx + dy*dy + dz*dz);
                        if (dist_cm > 20.0f) {
                            std::cout << u8"⛔ Position jump too large — ignoring\n";
                            goto show_only;
                        }
                    }

                    camera_positions.emplace_back(cx, cy, cz);
                    std::cout << "[Camera in tag frame] x=" << std::fixed << std::setprecision(2)
                              << cx << " y=" << cy << " z=" << cz << " cm\n";

                    draw_axes(frame, rvec, tvec, K, dist, corners[0]);

                    // (Optional) you could draw a simple 2D trail on the image here.

                    update_counter++;
                } else {
                    std::cout << u8"⚠️ Reprojection error too high (" << err << " px) — ignoring\n";
                }
            }
        }

    show_only:
        cv::imshow("Tag Localization", frame);
        int k = cv::waitKey(1);
        if ((k & 0xFF) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}

