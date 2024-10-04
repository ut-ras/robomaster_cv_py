# References:
# https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
# https://docs.opencv.org/4.x/d9/dc8/tutorial_py_trackbar.html
# https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
# https://www.opencv-srf.com/2010/09/object-detection-using-color-seperation.html
# https://docs.opencv.org/4.5.4/da/d97/tutorial_threshold_inRange.html
# https://docs.opencv.org/4.5.4/db/df6/tutorial_erosion_dilatation.html
# https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
# https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html
# https://stackoverflow.com/questions/38064777/use-waitkey-in-order-pause-and-play-video
# https://docs.opencv.org/4.x/de/d62/tutorial_bounding_rotated_ellipses.html

import cv2 as cv
import numpy as np
import time

# Measures how similar two numbers are
def sim(a, b):
    return min(a, b) / (a + b)

def get_contours(frame, color):
    # Apply a red mask to image, apply morphological opening/closing, and find contours of contiguous red areas
    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    mask1 = cv.inRange(frame_HSV, (0, 70, 50), (10, 255, 255))
    mask2 = cv.inRange(frame_HSV, (170, 70, 50), (180, 255, 255))

    frame_threshold = mask1 | mask2

    frame_threshold = cv.erode(frame_threshold, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
    frame_threshold = cv.dilate(frame_threshold, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))

    frame_threshold = cv.dilate(frame_threshold, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
    frame_threshold = cv.erode(frame_threshold, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))

    contours, _ = cv.findContours(frame_threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    return contours

def draw_centers(frame):
    contours = get_contours(frame)

    # Compute rotated bounding box for each contour and store in `bboxes`
    bboxes = []
    for contour in contours:
        bbox = cv.minAreaRect(contour)
        bboxes.append(bbox)
        bbox_points = cv.boxPoints(bbox)
        bbox_points = np.intp(bbox_points)
        frame = cv.drawContours(frame, [bbox_points], -1, (0, 255, 0), 2)

    thresh = 70
    width_sim_thresh, length_sim_thresh = 0.1, 0.4

    for i in range(len(bboxes)):
        bbox1 = bboxes[i]
        width1, length1, angle1 = bbox1[1][0], bbox1[1][1], bbox1[2]

        # If the bounding box is too small, skip
        if max(length1, width1) < thresh:
            continue

        # how the heck is `minAreaRect` defining the angle
        if angle1 > 45:
            angle1 = 90 - angle1
            width1, length1 = length1, width1

        # Matching longer sides is more important, and thus needs a stricter threshold
        if width1 > length1:
            width_sim_thresh, length_sim_thresh = length_sim_thresh, width_sim_thresh

        for j in range(i + 1, len(bboxes)):
            bbox2 = bboxes[j]
            width2, length2, angle2 = bbox2[1][0], bbox2[1][1], bbox2[2]

            if max(length2, width2) < thresh:
                continue

            if angle2 > 45:
                angle2 = 90 - angle2
                width2, length2 = length2, width2

            angle_diff = abs(angle1 - angle2)

            # If two bounding boxes are similar in size and orientation, place a dot between them
            if width_sim_thresh < sim(width1, width2) and length_sim_thresh < sim(length1, length2) and angle_diff < 5:
                cv.circle(frame, (round((bbox1[0][0] + bbox2[0][0]) / 2), round((bbox1[0][1] + bbox2[0][1]) / 2)), 10, (255, 0, 255), -1)

            # Debugging
            # print(f'w --- {i}: {width1}, {j}: {width2}, sim: {sim(width1, width2)}')
            # print(f'l --- {i}: {length1}, {j}: {length2}, sim: {sim(length1, length2)}')
            # print(f'a --- {i}: {angle1}, {j}: {angle2}, diff: {angle_diff}')

    # Reference square to see size of `thresh`
    # cv.rectangle(frame, (50, 50), (50 + thresh, 50 + thresh), (255, 0, 255), 1)

    return frame

def main():
    cap = cv.VideoCapture('IMG_2127.MOV')

    while cap.isOpened():
        # start_time = time.time()

        ret, frame = cap.read()

        if not ret:
            print('Failed to read frame. Exiting...')
            break

        frame = draw_centers(frame)

        cv.imshow('frame', frame)

        # end_time = time.time()
        # print(end_time - start_time)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

main()