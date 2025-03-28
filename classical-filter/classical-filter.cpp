#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

// Measures how similar two numbers are
double sim(double a, double b) {
    return min(a, b) / (a + b);
}

// Calculates the squared Euclidean distance between two points
double distSq(Point2f a, Point2f b) {
    return pow(a.x - b.x, 2) + pow(a.y - b.y, 2);
}

// Calculates the angle between two points in degrees
double angle(Point2f a, Point2f b) {
    if (a.x == b.x) return 90;
    Point2f right = (a.x > b.x) ? a : b;
    Point2f left = (a.x > b.x) ? b : a;
    return atan((right.y - left.y) / (right.x - left.x)) * 180 / CV_PI;
}

// Applies a mask to detect color-specific contours in the frame
vector<vector<Point>> getContours(Mat& frame, const string& color) {
    Mat frameHSV, mask1, mask2, frameThreshold;

    cvtColor(frame, frameHSV, COLOR_BGR2HSV);

    if (color == "red") {
        inRange(frameHSV, Scalar(0, 70, 50), Scalar(20, 255, 255), mask1);
        inRange(frameHSV, Scalar(170, 70, 50), Scalar(230, 255, 255), mask2);
    } else {
        inRange(frameHSV, Scalar(90, 70, 50), Scalar(120, 255, 255), mask1);
        inRange(frameHSV, Scalar(170, 70, 50), Scalar(200, 255, 255), mask2);
    }
    frameThreshold = mask1 | mask2;

    erode(frameThreshold, frameThreshold, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
    dilate(frameThreshold, frameThreshold, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
    dilate(frameThreshold, frameThreshold, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
    erode(frameThreshold, frameThreshold, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

    vector<vector<Point>> contours;
    findContours(frameThreshold, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    return contours;
}

// Draws the bounding box centers and processes relationships between them
Mat drawCenters(Mat frame, const string& color) {
    auto contours = getContours(frame, color);
    vector<RotatedRect> bboxes;

    for (auto& contour : contours) {
        RotatedRect bbox = minAreaRect(contour);
        bboxes.push_back(bbox);

        Point2f bboxPoints[4];
        bbox.points(bboxPoints);
        for (int j = 0; j < 4; j++) {
            line(frame, bboxPoints[j], bboxPoints[(j + 1) % 4], Scalar(0, 255, 0), 2);
        }
    }

    int thresh = 20;
    double widthSimThresh = 0.1, lengthSimThresh = 0.3, yThresh = 15, angleThresh = 15;

    for (size_t i = 0; i < bboxes.size(); i++) {
        auto& bbox1 = bboxes[i];
        Point2f center1 = bbox1.center;
        double width1 = bbox1.size.width;
        double length1 = bbox1.size.height;
        double angle1 = bbox1.angle;

        if (max(length1, width1) < thresh) continue;

        for (size_t j = i + 1; j < bboxes.size(); j++) {
            auto& bbox2 = bboxes[j];
            Point2f center2 = bbox2.center;
            double width2 = bbox2.size.width;
            double length2 = bbox2.size.height;
            double angle2 = bbox2.angle;

            if (max(length2, width2) < thresh) continue;

            double angleDiff = abs(angle1 - angle2);
            double yDiff = abs(center1.y - center2.y);

            if (sim(width1, width2) > widthSimThresh &&
                sim(length1, length2) > lengthSimThresh &&
                yDiff < yThresh &&
                (angleDiff < angleThresh || angleDiff > 180 - angleThresh)) {
                Point centerMid((center1.x + center2.x) / 2, (center1.y + center2.y) / 2);
                circle(frame, centerMid, 10, Scalar(255, 0, 255), -1);
            }
        }
    }

    return frame;
}

int main() {
    VideoCapture cap("tests/test1.mp4");

    if (!cap.isOpened()) {
        cout << "Error opening video file" << endl;
        return -1;
    }

    while (cap.isOpened()) {
        Mat frame;
        cap >> frame;

        if (frame.empty()) {
            cout << "End of video stream." << endl;
            break;
        }

        frame = drawCenters(frame, "blue");

        imshow("Frame", frame);

        if (waitKey(1) == 'q') break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
