#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

 // Blue mask
#define blueLowH 80
#define blueHighH 140
#define blueLowS 100 
#define blueHighS 255
#define blueLowV 175
#define blueHighV 255

// Red mask
#define redLowH_1 0
#define redHighH_1 10
#define redLowS_1 70 
#define redHighS_1 255
#define redLowV_1 50
#define redHighV_1 255

#define redLowH_2 170
#define redHighH_2 180
#define redLowS_2 70 
#define redHighS_2 255
#define redLowV_2 50
#define redHighV_2 255

using namespace cv;
using namespace std;

struct ArmorPlate {
    Point tl;
    Point br;
    Point center;
    RotatedRect left_light;
    RotatedRect right_light;
};

bool rect_sort_function (Rect first, Rect second)
{
    return first.tl().x < second.tl().x;
}

struct left_right_contour_sorter // 'less' for contours
{
    bool operator ()( const vector<Point>& a, const vector<Point> & b )
    {
        Rect ra(boundingRect(a));
        Rect rb(boundingRect(b));
        return (ra.x < rb.x);
    }
};

Mat mask_color(Mat hsv_img, bool red_mask) {
    Mat color_mask;
    if (red_mask) {
        Mat mask1, mask2;
        inRange(hsv_img, Scalar(redLowH_1, redLowS_1, redLowV_1), Scalar(redHighH_1, redHighS_1, redHighV_1), mask1);
        inRange(hsv_img, Scalar(redLowH_2, redLowS_2, redLowV_2), Scalar(redHighH_2, redHighS_2, redHighV_2), mask2);
        color_mask = mask1 | mask2;
    }
    else {
        inRange(hsv_img, Scalar(blueLowH, blueLowS, blueLowV), Scalar(blueHighH, blueHighS, blueHighV), color_mask); //Threshold the image
    }

    return color_mask;
}

Mat remove_artifacts(Mat img) {
    // morphologyEx(img, img, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
    // morphologyEx(img, img, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

    //morphological opening (removes small objects from the foreground)
    erode(img, img, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
    dilate(img, img, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 

    //morphological closing (removes small holes from the foreground)
    dilate(img, img, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 
    erode(img, img, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
    return img;
}

std::tuple<vector<Rect>, vector<vector<Point>>> find_bounding_boxes(Mat img) {
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    vector<Rect> accepted_rects;
    for (size_t i = 0; i < contours.size(); i++) // iterate through each contour.
    {
        Rect bounding_rect = boundingRect(contours[i]);
        if (bounding_rect.size().height / bounding_rect.size().width > 1
            && bounding_rect.size().height / bounding_rect.size().width < 7)
        {
            accepted_rects.push_back(bounding_rect);
        }
    }

    sort(accepted_rects.begin(), accepted_rects.end(), rect_sort_function);
    return std::make_tuple(accepted_rects, contours);
}

std::tuple<vector<RotatedRect>, vector<vector<Point>>> find_rotated_bounding_boxes(Mat img) {
    vector<vector<Point>> contours;
    findContours(img, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    sort(contours.begin(), contours.end(), left_right_contour_sorter());

    vector<RotatedRect> bounding_boxes;
    for (size_t i = 0; i < contours.size(); i++) {
        RotatedRect bounding_box = minAreaRect(contours[i]);
        Rect bounding_rect = bounding_box.boundingRect();
        if (bounding_rect.size().height / bounding_rect.size().width > 1 && bounding_rect.size().height / bounding_rect.size().width < 7)
            bounding_boxes.push_back(bounding_box);
    }

    return std::make_tuple(bounding_boxes, contours);
}

std::tuple<vector<ArmorPlate>, vector<Rect>> find_armor_plates(vector<Rect> bounding_boxes) {
    vector<ArmorPlate> armor_plates;
    if (bounding_boxes.size() >= 2)
    {
        Rect first = bounding_boxes[0];
        Rect second = bounding_boxes[1];

        int distance = abs(second.tl().y - first.tl().y);

        for (size_t i = 1; i + 1 < bounding_boxes.size(); i++)
        {
            if (abs(bounding_boxes[i].tl().y - bounding_boxes[i + 1].tl().y) < distance)
            {
                distance = abs(bounding_boxes[i].tl().y - bounding_boxes[i + 1].tl().y);
                first = bounding_boxes[i];
                second = bounding_boxes[i + 1];
            }
        }

        int x = (first.tl().x + second.br().x) / 2;
        int y = (first.tl().y + second.br().y) / 2;
        armor_plates.push_back(ArmorPlate{first.tl(), second.br(), Point(x,y)});
    }

    return std::make_tuple(armor_plates, bounding_boxes);
}

float normalize_0_180(RotatedRect bounding_box) {
    if (bounding_box.size.width < bounding_box.size.height) {
        return (90 - bounding_box.angle);
    }

    return -bounding_box.angle;
}

std::tuple<vector<ArmorPlate>, vector<RotatedRect>> find_rotated_armor_plates(vector<RotatedRect> bounding_boxes) {
    vector<ArmorPlate> armor_plates;
    if (bounding_boxes.size() >= 2) {
        for (size_t i = 0; i < bounding_boxes.size() - 1; i++) {
            RotatedRect first = bounding_boxes[i];
            RotatedRect second = bounding_boxes[i+1];

            Point2f first_points[4];
            Point2f second_points[4];
            first.points(first_points);
            second.points(second_points);

            int tl_x, tl_y, br_x, br_y;
            float angle_first = normalize_0_180(first);
            if (angle_first > 90) {
                tl_x = first_points[1].x;
                tl_y = min(first_points[1].y, second_points[2].y);
                br_x = second_points[3].x;
                br_y = max(first_points[0].y, second_points[3].y);
            }
            else {
                tl_x = first_points[3].x;
                tl_y = min(first_points[1].y, second_points[2].y);
                br_x = second_points[1].x;
                br_y = max(first_points[0].y, second_points[3].y);
            }

            Point tl(tl_x, tl_y), br(br_x, br_y);
            int x = (tl.x + br.x) / 2;
            int y = (tl.y + br.y) / 2;

            armor_plates.push_back(ArmorPlate{tl, br, Point(x,y), first, second});
        }
    }
    return std::make_tuple(armor_plates, bounding_boxes);
}

vector<ArmorPlate> find_best_armor_plates(vector<ArmorPlate> armor_plates) {
    int scores[armor_plates.size()];
    for (size_t i = 0; i < armor_plates.size(); i++) {
        scores[i] = 0;
    }

    // Check if armor plate light angles are within 5 degrees of each other
    for (size_t i = 0; i < armor_plates.size(); i++) {
        float angle_left = normalize_0_180(armor_plates.at(i).left_light);
        float angle_right = normalize_0_180(armor_plates.at(i).right_light);
        if (abs(angle_left - angle_right) < 5) {
            scores[i] += 1;
        }
    }

    // Check if armor plate light height differences are within 1/4 of light height
    for (size_t i = 0; i < armor_plates.size(); i++) {
        int left_light_y = armor_plates.at(i).left_light.boundingRect().tl().y;
        int right_light_y = armor_plates.at(i).right_light.boundingRect().tl().y;

        int avg_light_height = (armor_plates.at(i).left_light.boundingRect().height + armor_plates.at(i).right_light.boundingRect().height) / 2;

        if (abs(left_light_y - right_light_y) < avg_light_height / 4) {
            scores[i] += 1;
        }
    }

    // Check if armor plates is wider than it is tall
    for (size_t i = 0; i < armor_plates.size(); i++) {
        int height = armor_plates.at(i).br.y - armor_plates.at(i).tl.y;
        int width = armor_plates.at(i).br.x - armor_plates.at(i).tl.x;

        if (width > height) {
            scores[i] += 1;
        }
    }

    vector<ArmorPlate> best_armor_plates;
    bool finding_plates = true;
    while (finding_plates) {
        // Get next best score
        int max_score = -1;
        int max_index = -1;
        for (size_t i = 0; i < armor_plates.size(); i++) {
            if (scores[i] > max_score) {
                max_score = scores[i];
                max_index = i;
            }
        }

        // Remove neighbors and selected plate
        if (max_index > -1) {
            scores[max_index] = -1;
            if (max_index > 0) {
                scores[max_index - 1] = -1;
            }
            if (max_index < armor_plates.size() - 1) {
                scores[max_index + 1] = -1;
            }
            best_armor_plates.push_back(armor_plates.at(max_index));
        }

        // All plates found, exit
        else {
            finding_plates = false;
        }
    }

    return best_armor_plates;
}

int main( int argc, char** argv ) 
{
    while (true) 
    {
        bool isClosed = 0;
        VideoCapture cap("IMG_2127.MOV");//"../../resources/rmna2.mp4"); //xcapture the video from webcam

        if (!cap.isOpened() )  // if not success, exit program
        {
            cout << "Cannot open the web cam" << endl;
            return -1;
        }

        //Capture a temporary image from the camera
        Mat imgTmp;
        cap.read(imgTmp); 

        while (true) 
        {
            Mat img_original;
            bool bSuccess = cap.read(img_original); // read a new frame from video

            if (!bSuccess) //if not success, break loop
            {
                cout << "Cannot read a frame from video stream" << endl;
                break;
            }
            
            Mat hsv_img;
            cvtColor(img_original, hsv_img, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

            Mat color_mask = mask_color(hsv_img, true);
            color_mask = remove_artifacts(color_mask);

            Mat image_all_bounded_boxes = img_original.clone();

            auto [accepted_rects, contours] = find_rotated_bounding_boxes(color_mask);
            auto [armor_plates, bounding_boxes] = find_rotated_armor_plates(accepted_rects);
            
            armor_plates = find_best_armor_plates(armor_plates);
            for (int i = 0; i < armor_plates.size(); i++)
                circle(img_original, armor_plates[i].center, 2, Scalar(0, 0, 255), 8);

            for (size_t i = 0; i < bounding_boxes.size(); i++) // iterate through each contour.
            {
                Rect brect = bounding_boxes[i].boundingRect();
                rectangle(img_original, brect, Scalar(0, 255, 0), 5); 
            }

            imshow("Original", img_original); //show the modified image

            if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
            {
                cout << "esc key is pressed by user" << endl;
                isClosed = 1;
                break; 
            }

            if (isClosed) 
            {
                break;
            }
        }
    }
    
    return 0;
}