import numpy as np
import cv2
import copy
import time

def contour_generator(frame):
    test_img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    test_blur = cv2.GaussianBlur(test_img1, (7, 7), 0)
    edge = cv2.Canny(test_blur, 75, 175)
    edge1 = copy.copy(edge)
    contour_list = list()

    cnts, h = cv2.findContours(edge1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # error handling
    
    if h is None:
        return []

    index = list()
    for hier in h[0]:
        if hier[3] != -1:
            index.append(hier[3])

    # loop over the contours
    for c in index:
        peri = cv2.arcLength(cnts[c], True)
        approx = cv2.approxPolyDP(cnts[c], 0.02 * peri, True)

        if len(approx) > 4:
            peri1 = cv2.arcLength(cnts[c - 1], True)
            corners = cv2.approxPolyDP(cnts[c - 1], 0.02 * peri1, True)
            contour_list.append(corners)

    new_contour_list = list()
    for contour in contour_list:
        if len(contour) == 4:
            new_contour_list.append(contour)
    final_contour_list = list()
    for element in new_contour_list:
        if cv2.contourArea(element) > 700 and cv2.isContourConvex(element):
            final_contour_list.append(element)

    return final_contour_list

# Function to return the order of points in camera frame
def order(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    # print(np.argmax(s))
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    # print(np.argmax(diff))
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

lowerRed1 = np.array([0, 50, 50])
upperRed1 = np.array([10, 255, 255])
lowerRed2 = np.array([170, 50, 50])
upperRed2 = np.array([180, 255, 255])

lowerBlue = np.array([100,80,160])
upperBlue = np.array([120,255,255])

def determineColor(marker):
    markerHSV = cv2.cvtColor(marker, cv2.COLOR_BGR2HSV)

    redMask1 = cv2.inRange(markerHSV, lowerRed1, upperRed1)
    redMask2 = cv2.inRange(markerHSV, lowerRed2, upperRed2)
    redMask = redMask1 + redMask2

    blueMask = cv2.inRange(markerHSV, lowerBlue, upperBlue)

    cv2.imshow("redMask1", redMask1)
    cv2.imshow("redMask2", redMask1)
    cv2.imshow("redMask", redMask)
    cv2.imshow("blueMask", blueMask)

    if np.sum(redMask) > 0:
        return "red"
    elif np.sum(blueMask) > 0:
        return "blue"
    return "unknown"

def determineLetter(marker):
    ret, img_bw = cv2.threshold(marker, 200, 255, cv2.THRESH_BINARY)
    white = 255
    cropped_img = img_bw[25:150, 25:150]
    cv2.imshow("Cropped", cropped_img)

    # [y, x]: [0, 0] = top left pixel
    if cropped_img[12, 37] != white:    # Center of pixel at [0, 1]; not white if letter "A"
        return "A"
    elif cropped_img[12, 12] != white:  # Center of pixel at [0, 0]; not white if letter "B"
        return "C"
    elif cropped_img[12, 112] == white: # Center of pixel at [0, 4]; white if letter "E"
        return "E"
    elif cropped_img[62, 112] == white: # Center of pixel at [2, 4]; white if letter "D"
        return "D"
    else:
        return "B"                      # Must be "B" if none of the above



fid_size = 0.053  # meters
# TODO fix calibration to setup cameraMatrix
cameraMatrix = np.array([[1.25649815e+03, 0.0, 7.12996774e+02],
                         [0.0, 1.25820533e+03, 4.69551858e+02], 
                         [0.0, 0.0, 1.0]], dtype=np.float32)
distCoeffs = np.array([[-3.72271817e-03, 5.33786890e-01, -4.99625728e-04, -1.65101232e-03,-1.78505927e+00]], dtype=np.float32)
# TODO testing program with hardcoded matrix since calibration session is not working
# TODO when game ready, dont forget to comment this out

def findTranslationAndRotation(h):
    object_points = np.array([[-fid_size / 2.0, fid_size / 2.0, 0.0],
                              [fid_size / 2.0, fid_size / 2.0, 0.0],
                              [fid_size / 2.0, -fid_size / 2.0, 0.0],
                              [-fid_size / 2.0, -fid_size / 2.0, 0.0]], dtype=np.float32)
    # TODO error here since cameraMatrix is null
    _, rvec, tvec = cv2.solvePnP(object_points, np.array(h, dtype=np.float32), cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)

    if rvec is not None:
        Rt = np.matrix(cv2.Rodrigues(rvec)[0])
        yaw, pitch, roll = rotToEul(Rt)
        R = Rt.T
        pose = -R * np.matrix(tvec)

        print("CAMERA Position x=" + str(pose[2]) + " y=" + str(pose[0]) + " z=" + str(pose[1]))
        print("Camera tilt -------   yaw=" + str(yaw) + " pitch=" + str(pitch) + " roll=" + str(roll))
        return pose, (yaw, pitch, roll)
    
    return "ERROR: Rotation Vector Not Found"

def rotToEul(R):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    singular = sy < 1e-6
    if not singular:
        yaw = np.arctan2(R[2, 1], R[2, 2])  # Rotation around Z-axis
        pitch = np.arctan2(-R[2, 0], sy)    # Rotation around Y-axis
        roll = np.arctan2(R[1, 0], R[0, 0]) # Rotation around X-axis
    else:
        yaw = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        roll = 0

    # Convert radians to degrees
    return np.degrees([yaw, pitch, roll])

cap = cv2.VideoCapture(0)

prev_time = 0
new_time = 0

dim = 175
p1 = np.array([
    [0, 0],
    [dim - 1, 0],
    [dim - 1, dim - 1],
    [0, dim - 1]], dtype="float32")

while(True):
    ret, frame = cap.read()
    if ret == True:
        final_contour_list = contour_generator(frame)
        for i in range(len(final_contour_list)):
            x, y, w, h = cv2.boundingRect(final_contour_list[i])
            c_rez = final_contour_list[i][:, 0]
            h, _ = cv2.findHomography(order(c_rez), p1, cv2.RANSAC, 2)

            if h is not None:
                tag = cv2.warpPerspective(frame, h, (175, 175))
                markerColor = determineColor(tag)
                markerLetter = determineLetter(cv2.cvtColor(tag, cv2.COLOR_BGR2GRAY))
                tvec, angles = findTranslationAndRotation(order(c_rez)) # error here
                print("Tag: " + markerLetter + ", Color: " + markerColor)
                # TODO find the distance of the camera from the tag
                # TODO determine the coordinate of the specific tag by looking it up through a hard coded map of each tag's map coordinates
                # TODO determine the angle from the mid point at which the tag exists
                # TODO create an algorithm that uses distance, angle, and tag coordinates to determine position
                # TODO update low-level positioning data

                cv2.drawContours(frame, [final_contour_list[i]], -1, (0, 255, 0), 2)
                cv2.putText(frame, text=str(cv2.contourArea(final_contour_list[i])), org=(x, y-5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.25, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
                cv2.imshow("Marker", tag)

        new_time = time.time()
        fps = 1/(new_time - prev_time)
        prev_time = new_time
        fps = int(fps)
        fps = str(fps)
        cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (1000, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow("Outline", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()