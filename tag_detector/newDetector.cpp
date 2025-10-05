// tag_localization_dual.cpp
// Build (Ubuntu):
//   g++ -std=c++17 tag_localization_dual.cpp -o tag_loc `pkg-config --cflags --libs opencv4`
// Run:
//   ./tag_loc
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <cmath>
#include <iomanip>

using namespace cv;
using std::cout;
using std::endl;

// ----------------- Config / constants -----------------
static const float fid_size_mm = 100.0f; // mm
// Camera intrinsics (hardcoded like your Python)
static const Mat cameraMatrix = (Mat_<float>(3,3) <<
    1.18666112e+03f, 0.f, 7.22383627e+02f,
    0.f, 1.19064020e+03f, 4.94566994e+02f,
    0.f, 0.f, 1.f);
// Zero distortion for testing (like your file)
static const Mat distCoeffs = Mat::zeros(1, 5, CV_32F);

// Field tag poses (x, y, directionDeg)
static const std::map<std::string, cv::Vec3f> TAG_POSES = {
    {"A_RD",{  500.f, 7999.f,  90.f}},
    {"B_RD",{    0.f,  500.f, 180.f}},
    {"C_RD",{ 2999.f, 3500.f,   0.f}},
    {"D_RD",{ 3171.f, 5500.f, 180.f}},
    {"E_RD",{ 5350.f, 7999.f,  90.f}},
    {"A_BL",{11500.f,    0.f, 270.f}},
    {"B_BL",{12000.f, 7500.f,   0.f}},
    {"C_BL",{ 9001.f, 4500.f, 180.f}},
    {"D_BL",{ 8829.3f,2500.f,   0.f}},
    {"E_BL",{ 6750.f,    0.f, 270.f}},
};

// ----------------- Helpers -----------------
static std::vector<Point2f> order_corners(const std::vector<Point2f>& pts) {
    // Same logic as Python 'order': uses sum and diff to assign TL, TR, BR, BL
    CV_Assert(pts.size() == 4);
    std::vector<Point2f> rect(4);
    std::vector<float> sumv, diffv;
    sumv.reserve(4); diffv.reserve(4);
    for (const auto& p : pts) {
        sumv.push_back(p.x + p.y);
        diffv.push_back(p.y - p.x);
    }
    // TL: smallest sum, BR: largest sum
    int i_tl = int(std::distance(sumv.begin(), std::min_element(sumv.begin(), sumv.end())));
    int i_br = int(std::distance(sumv.begin(), std::max_element(sumv.begin(), sumv.end())));
    // TR: smallest diff, BL: largest diff
    int i_tr = int(std::distance(diffv.begin(), std::min_element(diffv.begin(), diffv.end())));
    int i_bl = int(std::distance(diffv.begin(), std::max_element(diffv.begin(), diffv.end())));
    rect[0] = pts[i_tl];
    rect[1] = pts[i_tr];
    rect[2] = pts[i_br];
    rect[3] = pts[i_bl];
    return rect;
}

static std::pair<Mat, Mat> detect_target(const Mat& frame_bgr) {
    // Returns two masks (for debug) but we’ll actually return contours lists in main
    Mat hsv; cvtColor(frame_bgr, hsv, COLOR_BGR2HSV);

    // Red ranges
    Mat mask1, mask2, maskRed;
    inRange(hsv, Scalar(0,120,70),   Scalar(10,255,255),  mask1);
    inRange(hsv, Scalar(170,120,70), Scalar(180,255,255), mask2);
    maskRed = mask1 | mask2;

    // Blue range
    Mat maskBlue;
    inRange(hsv, Scalar(90,80,160), Scalar(120,255,255), maskBlue);
    return {maskRed, maskBlue};
}

static std::vector<std::vector<Point>> contours_from_mask(const Mat& mask) {
    Mat red_regions; // not strictly needed to find edges; but follow Python’s idea
    // Use mask directly for edges to reduce extra ops
    Mat edges; Canny(mask, edges, 50, 150);

    std::vector<std::vector<Point>> contours;
    findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    std::sort(contours.begin(), contours.end(),
              [](auto& a, auto& b){ return contourArea(a) > contourArea(b); });
    return contours;
}

static std::vector<std::vector<Point2f>> quadrilateral_candidates(const std::vector<std::vector<Point>>& contours,
                                                                  Mat& frame_to_draw,
                                                                  double min_area = 500.0) {
    std::vector<std::vector<Point2f>> quads;
    for (const auto& c : contours) {
        double peri = arcLength(c, true);
        std::vector<Point> approx;
        approxPolyDP(c, approx, 0.02 * peri, true);
        if (approx.size() == 4) {
            double area = contourArea(approx);
            if (area < min_area) continue;
            if (!isContourConvex(approx)) continue;

            std::vector<Point2f> corners;
            corners.reserve(4);
            for (const auto& p : approx) corners.emplace_back((float)p.x, (float)p.y);

            // draw for visualization
            std::vector<std::vector<Point>> drawPoly(1);
            for (auto& q : approx) drawPoly[0].push_back(q);
            drawContours(frame_to_draw, drawPoly, -1, Scalar(0,255,0), 3);

            quads.push_back(corners);
        }
    }
    return quads;
}

// returns "A","B","C","D","E", or empty string if none
static std::string determineLetter(const Mat& markerGray) {
    Mat img_bw;
    threshold(markerGray, img_bw, 200, 255, THRESH_BINARY);
    const uchar white = 255;

    // mimic Python indices
    if (img_bw.at<uchar>(5,5) == white) {
        return ""; // False / not a valid tag
    }

    Mat cropped = img_bw(Rect(25, 25, 125, 125)); // [25:150, 25:150]
    // Access [y,x] in cropped
    auto px = [&](int y, int x)->uchar { return cropped.at<uchar>(y, x); };

    if (px(12, 37) != white && px(12, 62) == white) {
        imshow("Cropped", cropped);
        return "A";
    } else if (px(12, 12) != white && px(12, 37) == white) {
        imshow("Cropped", cropped);
        return "C";
    } else if (px(37, 37) != white && px(12, 112) == white) {
        imshow("Cropped", cropped);
        return "E";
    } else if (px(12, 112) != white && px(62, 112) == white) {
        imshow("Cropped", cropped);
        return "D";
    } else if (px(37, 112) == white && px(62, 122) != white) {
        imshow("Cropped", cropped);
        return "B";
    }
    return "";
}

// Optional: convert rotation matrix to yaw/pitch/roll (deg)
static cv::Vec3d rotToEul(const Mat& R) {
    // R is 3x3 (double)
    double sy = std::sqrt(R.at<double>(0,0)*R.at<double>(0,0) + R.at<double>(1,0)*R.at<double>(1,0));
    bool singular = sy < 1e-6;

    double yaw, pitch, roll;
    if (!singular) {
        yaw   = std::atan2(R.at<double>(2,1), R.at<double>(2,2)); // Z
        pitch = std::atan2(-R.at<double>(2,0), sy);               // Y
        roll  = std::atan2(R.at<double>(1,0), R.at<double>(0,0)); // X
    } else {
        yaw   = std::atan2(-R.at<double>(1,2), R.at<double>(1,1));
        pitch = std::atan2(-R.at<double>(2,0), sy);
        roll  = 0.0;
    }
    return { yaw * 180.0/M_PI, pitch * 180.0/M_PI, roll * 180.0/M_PI };
}

static bool findTranslationAndRotation(const std::vector<Point2f>& image_pts,
                                       cv::Vec3d& tvec, cv::Vec3d& rpy_deg)
{
    std::vector<Point3f> obj = {
        { -fid_size_mm/2.f,  fid_size_mm/2.f, 0.f },
        {  fid_size_mm/2.f,  fid_size_mm/2.f, 0.f },
        {  fid_size_mm/2.f, -fid_size_mm/2.f, 0.f },
        { -fid_size_mm/2.f, -fid_size_mm/2.f, 0.f }
    };
    Mat rvec, tvecm;
    bool ok = solvePnP(obj, image_pts, cameraMatrix, distCoeffs, rvec, tvecm, false, SOLVEPNP_IPPE);
    if (!ok) return false;

    Mat R;
    Rodrigues(rvec, R);           // R: 3x3, double by default
    cv::Vec3d rpy = rotToEul(R);  // yaw, pitch, roll in deg

    tvec = cv::Vec3d(tvecm.at<double>(0,0), tvecm.at<double>(1,0), tvecm.at<double>(2,0));
    rpy_deg = rpy;
    return true;
}

// ----------------- Main -----------------
int main() {
    std::cout.setf(std::ios::fixed); std::cout << std::setprecision(2);

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open webcam.\n";
        return 1;
    }

    const int dim = 175;
    std::vector<Point2f> p1 = {
        {0.f, 0.f},
        {(float)(dim-1), 0.f},
        {(float)(dim-1), (float)(dim-1)},
        {0.f, (float)(dim-1)}
    };

    Mat frame;
    while (true) {
        if (!cap.read(frame) || frame.empty()) break;

        // Detect red & blue masks, then contours → quads
        Mat maskRed, maskBlue;
        std::tie(maskRed, maskBlue) = detect_target(frame);

        auto contoursR = contours_from_mask(maskRed);
        auto contoursB = contours_from_mask(maskBlue);

        // For visualization, draw accepted quads on the original frame
        auto quadsR = quadrilateral_candidates(contoursR, frame);
        auto quadsB = quadrilateral_candidates(contoursB, frame);

        // Process RED quads
        for (const auto& corners : quadsR) {
            // Order corners (TL,TR,BR,BL)
            auto ordered = order_corners(corners);

            // Homography and warp to 175x175
            Mat H = findHomography(ordered, p1, RANSAC, 2.0);
            if (H.empty()) continue;

            Mat tag; warpPerspective(frame, tag, H, Size(dim, dim));
            Mat tagGray; cvtColor(tag, tagGray, COLOR_BGR2GRAY);

            std::string letter = determineLetter(tagGray);
            if (letter.empty()) continue;

            cv::Vec3d tvec, rpy_deg;
            if (findTranslationAndRotation(ordered, tvec, rpy_deg)) {
                cout << "Red Tag: " << letter << "\n";
                // Camera position print in their mapping: x=tvec[2], y=tvec[0], z=tvec[1]
                cout << "CAMERA Position x=" << tvec[2] << " y=" << tvec[0] << " z=" << tvec[1] << "\n";
                cout << "Camera tilt -------   yaw=" << rpy_deg[0]
                     << " pitch=" << rpy_deg[1]
                     << " roll="  << rpy_deg[2] << "\n";

                std::string tagmapname = letter + "_RD";
                auto it = TAG_POSES.find(tagmapname);
                if (it != TAG_POSES.end()) {
                    float base_dir = it->second[2];
                    cout << "2D Pose Angle -----   " << (base_dir + (float)rpy_deg[1]) << " degrees\n";

                    float fx = it->second[0];
                    float fy = it->second[1];
                    if (base_dir == 0.f) {
                        cout << "Field Position ----   x=" << (fx - tvec[2]) << " y=" << (fy + tvec[0]) << "\n";
                    } else if (base_dir == 90.f) {
                        cout << "Field Position ----   x=" << (fx - tvec[0]) << " y=" << (fy - tvec[2]) << "\n";
                    } else if (base_dir == 180.f) {
                        cout << "Field Position ----   x=" << (fx + tvec[2]) << " y=" << (fy - tvec[0]) << "\n";
                    } else {
                        cout << "Field Position ----   x=" << (fx + tvec[0]) << " y=" << (fy + tvec[2]) << "\n";
                    }
                }
            }
        }

        // Process BLUE quads
        for (const auto& corners : quadsB) {
            auto ordered = order_corners(corners);
            Mat H = findHomography(ordered, p1, RANSAC, 2.0);
            if (H.empty()) continue;

            Mat tag; warpPerspective(frame, tag, Size(dim, dim), H);
            Mat tagGray; cvtColor(tag, tagGray, COLOR_BGR2GRAY);

            std::string letter = determineLetter(tagGray);
            if (letter.empty()) continue;

            cv::Vec3d tvec, rpy_deg;
            if (findTranslationAndRotation(ordered, tvec, rpy_deg)) {
                cout << "Blue Tag: " << letter << "\n";
                cout << "CAMERA Position x=" << tvec[2] << " y=" << tvec[0] << " z=" << tvec[1] << "\n";
                cout << "Camera tilt -------   yaw=" << rpy_deg[0]
                     << " pitch=" << rpy_deg[1]
                     << " roll="  << rpy_deg[2] << "\n";

                std::string tagmapname = letter + "_BL";
                auto it = TAG_POSES.find(tagmapname);
                if (it != TAG_POSES.end()) {
                    float base_dir = it->second[2];
                    cout << "2D Pose Angle -----   " << (base_dir + (float)rpy_deg[1]) << " degrees\n";

                    float fx = it->second[0];
                    float fy = it->second[1];
                    if (base_dir == 0.f) {
                        cout << "Field Position ----   x=" << (fx - tvec[2]) << " y=" << (fy + tvec[0]) << "\n";
                    } else if (base_dir == 90.f) {
                        cout << "Field Position ----   x=" << (fx - tvec[0]) << " y=" << (fy - tvec[2]) << "\n";
                    } else if (base_dir == 180.f) {
                        cout << "Field Position ----   x=" << (fx + tvec[2]) << " y=" << (fy - tvec[0]) << "\n";
                    } else {
                        cout << "Field Position ----   x=" << (fx + tvec[0]) << " y=" << (fy + tvec[2]) << "\n";
                    }
                }
            }
        }

        imshow("Webcam Detection", frame);
        int k = waitKey(1);
        if ((k & 0xFF) == 'q') break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}

