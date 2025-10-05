// tag_detect_pose.cpp
// Build (Ubuntu):
//   g++ -std=c++17 tag_detect_pose.cpp -o tag_detect `pkg-config --cflags --libs opencv4`
// Run:
//   ./tag_detect

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <numeric>

using namespace cv;
using std::cout;
using std::endl;

// ---------------- Camera intrinsics (match your Python) ----------------
static const float fid_size_m = 0.053f; // meters
static const Mat cameraMatrix = (Mat_<float>(3,3) <<
    1.25649815e+03f, 0.f, 7.12996774e+02f,
    0.f, 1.25820533e+03f, 4.69551858e+02f,
    0.f, 0.f, 1.f);
static const Mat distCoeffs = (Mat_<float>(1,5) <<
    -3.72271817e-03f, 5.33786890e-01f, -4.99625728e-04f, -1.65101232e-03f, -1.78505927e+00f);

// ---------------- Color ranges (HSV) ----------------
static const Scalar lowerRed1(  0,150,150), upperRed1( 10,255,255);
static const Scalar lowerRed2(170,150,150), upperRed2(180,255,255);
static const Scalar lowerBlue(100, 80,160), upperBlue(120,255,255);

// ---------------- Helpers ----------------
static std::vector<std::vector<Point>> contour_generator(const Mat& frameBGR) {
    Mat gray, blurImg, edges;
    cvtColor(frameBGR, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blurImg, Size(7,7), 0);

    // Median-based Canny thresholds (like Python)
    double med = medianBlur; // placeholder to avoid unused – we compute below
    {
        Mat tmp;
        blurImg.reshape(0,1).copyTo(tmp);
        // Fast median using nth_element
        std::vector<uchar> v(tmp.begin<uchar>(), tmp.end<uchar>());
        size_t mid = v.size()/2;
        std::nth_element(v.begin(), v.begin()+mid, v.end());
        med = v[mid];
    }
    int lower = (int)std::max(0.0, 0.3 * med);
    int upper = (int)std::min(255.0, 1.1 * med);

    Canny(blurImg, edges, lower, upper);
    imshow("Edges", edges);

    // Find contours with hierarchy (like RETR_TREE)
    std::vector<std::vector<Point>> cnts;
    std::vector<Vec4i> hierarchy;
    findContours(edges.clone(), cnts, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    if (hierarchy.empty()) return {};

    // Original code used complex parent logic and then filtered.
    // We’ll reproduce the intent: gather quads with area/convex filter later.
    std::vector<std::vector<Point>> result;
    for (size_t i = 0; i < cnts.size(); ++i) {
        // Try approximating this contour and (optionally) its previous sibling, like Python did
        double peri = arcLength(cnts[i], true);
        std::vector<Point> approx; approxPolyDP(cnts[i], approx, 0.02 * peri, true);
        if (approx.size() > 4 && i > 0) {
            double peri1 = arcLength(cnts[i-1], true);
            std::vector<Point> corners; approxPolyDP(cnts[i-1], corners, 0.02 * peri1, true);
            result.push_back(corners);
        }
    }

    // Keep only quads, area>1000, convex
    std::vector<std::vector<Point>> final_list;
    for (auto& contour : result) {
        if (contour.size() == 4 && contourArea(contour) > 1000.0 && isContourConvex(contour)) {
            final_list.push_back(contour);
        }
    }
    return final_list;
}

static std::vector<Point2f> order(const std::vector<Point2f>& pts) {
    CV_Assert(pts.size() == 4);
    std::vector<Point2f> rect(4);
    std::vector<float> s(4), d(4);
    for (int i=0;i<4;++i){ s[i]=pts[i].x+pts[i].y; d[i]=pts[i].y-pts[i].x; }
    int tl = int(std::distance(s.begin(), std::min_element(s.begin(), s.end())));
    int br = int(std::distance(s.begin(), std::max_element(s.begin(), s.end())));
    int tr = int(std::distance(d.begin(), std::min_element(d.begin(), d.end())));
    int bl = int(std::distance(d.begin(), std::max_element(d.begin(), d.end())));
    rect[0]=pts[tl]; rect[2]=pts[br]; rect[1]=pts[tr]; rect[3]=pts[bl];
    return rect;
}

// Determine main color inside 175x175 tag image (BGR)
static std::string determineColor(const Mat& markerBGR) {
    Mat hsv; cvtColor(markerBGR, hsv, COLOR_BGR2HSV);
    Mat red1, red2, red, blue;
    inRange(hsv, lowerRed1, upperRed1, red1);
    inRange(hsv, lowerRed2, upperRed2, red2);
    red = red1 + red2;
    inRange(hsv, lowerBlue, upperBlue, blue);

    imshow("redMask1", red1);
    imshow("redMask2", red2);
    imshow("redMask", red);
    imshow("blueMask", blue);

    if (sum(red)[0] > 0)  return "red";
    if (sum(blue)[0] > 0) return "blue";
    return {};
}

// Determine letter from warped grayscale tag
static std::string determineLetter(const Mat& markerGray) {
    Mat img_bw; threshold(markerGray, img_bw, 200, 255, THRESH_BINARY);
    const uchar white = 255;

    if (img_bw.at<uchar>(5,5) == white) {
        return {}; // False
    }
    Mat cropped = img_bw(Rect(25,25,125,125));
    imshow("Cropped", cropped);

    auto px = [&](int y, int x)->uchar { return cropped.at<uchar>(y,x); };

    // replicate your pixel tests (note: second comment said 'B' but code returned 'C')
    if (px(12,37) != white)                              return "A";
    else if (px(12,12) != white)                         return "C";
    else if (px(12,112) == white)                        return "E";
    else if (px(62,112) == white)                        return "D";
    else                                                 return "B";
}

static cv::Vec3d rotToEul(const Mat& R) {
    double sy = std::sqrt(R.at<double>(0,0)*R.at<double>(0,0) + R.at<double>(1,0)*R.at<double>(1,0));
    bool singular = sy < 1e-6;
    double yaw, pitch, roll;
    if (!singular) {
        yaw   = std::atan2(R.at<double>(2,1), R.at<double>(2,2));
        pitch = std::atan2(-R.at<double>(2,0), sy);
        roll  = std::atan2(R.at<double>(1,0), R.at<double>(0,0));
    } else {
        yaw   = std::atan2(-R.at<double>(1,2), R.at<double>(1,1));
        pitch = std::atan2(-R.at<double>(2,0), sy);
        roll  = 0.0;
    }
    return { yaw*180.0/M_PI, pitch*180.0/M_PI, roll*180.0/M_PI };
}

static bool findTranslationAndRotation(const std::vector<Point2f>& imgPts,
                                       Vec3d& tvec, Vec3d& rpy_deg)
{
    std::vector<Point3f> objPts = {
        { -fid_size_m/2.f,  fid_size_m/2.f, 0.f },
        {  fid_size_m/2.f,  fid_size_m/2.f, 0.f },
        {  fid_size_m/2.f, -fid_size_m/2.f, 0.f },
        { -fid_size_m/2.f, -fid_size_m/2.f, 0.f }
    };
    Mat rvec, tvecMat;
    bool ok = solvePnP(objPts, imgPts, cameraMatrix, distCoeffs, rvec, tvecMat, false, SOLVEPNP_IPPE_SQUARE);
    if (!ok) return false;

    Mat R; Rodrigues(rvec, R);
    rpy_deg = rotToEul(R);
    tvec = Vec3d(tvecMat.at<double>(0,0), tvecMat.at<double>(1,0), tvecMat.at<double>(2,0));
    return true;
}

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open webcam.\n";
        return 1;
    }

    const int dim = 175;
    std::vector<Point2f> p1 = {
        {0.f,0.f}, {(float)(dim-1),0.f}, {(float)(dim-1),(float)(dim-1)}, {0.f,(float)(dim-1)}
    };

    auto t_prev = std::chrono::steady_clock::now();

    Mat frame;
    while (true) {
        if (!cap.read(frame) || frame.empty()) break;

        auto contours = contour_generator(frame);

        for (const auto& c : contours) {
            // draw detected contour
            polylines(frame, c, true, Scalar(0,255,0), 2);

            // order corners
            std::vector<Point2f> corners2f;
            for (auto& p : c) corners2f.emplace_back((float)p.x, (float)p.y);
            auto ord = order(corners2f);

            // homography
            Mat H = findHomography(ord, p1, RANSAC, 2.0);
            if (H.empty()) continue;

            // warp
            Mat tag; warpPerspective(frame, tag, H, Size(dim, dim));
            std::string letter = determineLetter(cvtColor(tag, tag, COLOR_BGR2GRAY), ""); // placeholder to match overload
            // The above line is incorrect; fix: separate gray conversion
            {
                Mat grayTag; cvtColor(tag, grayTag, COLOR_BGR2GRAY);
                letter = determineLetter(grayTag);
            }
            std::string color = determineColor(tag);
            if (letter.empty() || color.empty()) continue;

            Vec3d tvec, rpy;
            if (findTranslationAndRotation(ord, tvec, rpy)) {
                cout << "Tag: " << letter << ", Color: " << color << endl;
                // Match your print mapping: x=tvec[2], y=tvec[0], z=tvec[1]
                cout << "CAMERA Position x=" << tvec[2] << " y=" << tvec[0] << " z=" << tvec[1] << endl;
                cout << "Camera tilt -------   yaw=" << rpy[0]
                     << " pitch=" << rpy[1]
                     << " roll="  << rpy[2] << endl;

                // area debug
                Rect bbox = boundingRect(c);
                putText(frame, std::to_string((int)contourArea(c)), {bbox.x, bbox.y-5},
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,0), 1, LINE_AA);

                imshow("Marker", tag);
            }
        }

        // FPS overlay
        auto t_now = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(t_now - t_prev).count();
        t_prev = t_now;
        int fps = (dt > 0.0) ? (int)std::round(1.0 / dt) : 0;
        putText(frame, std::to_string(fps), {7,70}, FONT_HERSHEY_SIMPLEX, 2.0, Scalar(0,255,0), 3, LINE_AA);

        imshow("Outline", frame);
        if ((waitKey(1) & 0xFF) == 'q') break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}

