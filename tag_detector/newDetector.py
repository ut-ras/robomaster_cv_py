import cv2
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.set_xlim([-1000, 1000])  # testing with -1 to 1 meter range for each axis
ax.set_ylim([-1000, 1000])
ax.set_zlim([-1000, 1000])

point, = ax.plot([], [], [], 'ro', markersize=8)
orig, = ax.plot([], [], [], 'bo', markersize=8)

def update_3d_plot(tvec):
    if tvec is not None:
        x = tvec[2].item()
        y = tvec[0].item()
        z = tvec[1].item()
        point.set_data([x], [y])
        point.set_3d_properties([z])
        orig.set_data([0], [0])
        orig.set_3d_properties([0])
        plt.draw()
        plt.pause(0.01)

def detect_target(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define red color range in HSV
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    lowerBlue = np.array([90,80,160])
    upperBlue = np.array([120,255,255])
    
    # Create masks to detect red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    maskRed = mask1 + mask2

    maskBlue = cv2.inRange(hsv, lowerBlue, upperBlue)
    
    # Apply mask to extract red regions
    red_regions = cv2.bitwise_and(frame, frame, mask=maskRed)
    blue_regions = cv2.bitwise_and(frame, frame, mask = maskBlue)

    grayRed = cv2.cvtColor(red_regions, cv2.COLOR_BGR2GRAY)
    grayBlue = cv2.cvtColor(blue_regions, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("blue", maskBlue)
    
    edgesRed = cv2.Canny(grayRed, 50, 150)
    edgesB = cv2.Canny(grayBlue, 50, 150)
    # cv2.imshow("edges", edgesB)
    contoursRed, _ = cv2.findContours(edgesRed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contoursB, _ = cv2.findContours(edgesB, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contoursRed = sorted(contoursRed, key=cv2.contourArea, reverse=True)
    contoursB = sorted(contoursB, key=cv2.contourArea, reverse=True)
    
    approx_cornersRed = None
    approx_cornersB = None
    min_area_threshold = 500  # Ignore small detections

    # Do edge filtering for both Red and Blue contours, return all contours that fit criteria
    contour_listR = list()
    for contourRed in contoursRed:
        epsilon = 0.02 * cv2.arcLength(contourRed, True)
        approxRed = cv2.approxPolyDP(contourRed, epsilon, True)
        
        if len(approxRed) == 4:
            areaRed = cv2.contourArea(approxRed)
            if areaRed < min_area_threshold:
                continue  # Ignore small contours
            
            # Convexity check
            if not cv2.isContourConvex(approxRed):
                continue
            
            approx_cornersRed = approxRed
            if approx_cornersRed is not None:
                cv2.drawContours(frame, [approx_cornersRed], -1, (0, 255, 0), 3)
                contour_listR.append(approx_cornersRed.reshape(4, 1, 2))
            
    contour_listB = list()
    for contourB in contoursB:
        epsilon = 0.02 * cv2.arcLength(contourB, True)
        approxB = cv2.approxPolyDP(contourB, epsilon, True)
        
        if len(approxB) == 4:
            areaB = cv2.contourArea(approxB)
            if areaB < min_area_threshold:
                continue  # Ignore small contours
            
            # Convexity check
            if not cv2.isContourConvex(approxB):
                continue
            
            approx_cornersB = approxB
            if approx_cornersB is not None:
                cv2.drawContours(frame, [approx_cornersB], -1, (0, 255, 0), 3)
                contour_listB.append(approx_cornersB.reshape(4, 1, 2))
        
    return frame, contour_listR, contour_listB

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

def determineLetter(marker):
    ret, img_bw = cv2.threshold(marker, 200, 255, cv2.THRESH_BINARY)
    white = 255
    if img_bw[5, 5] == white:
        return False
    cropped_img = img_bw[25:150, 25:150]

    # [y, x]: [0, 0] = top left pixel
    # Each pixel seems to be like 25 
    if cropped_img[12, 37] != white and cropped_img[12, 62] == white:    # Center of pixel at [0, 1]; not white if letter "A"
        cv2.imshow("Cropped", cropped_img)
        return "A"
    elif cropped_img[12, 12] != white and cropped_img[12, 37] == white:  # Center of pixel at [0, 0]; not white if letter "C"
        cv2.imshow("Cropped", cropped_img)
        return "C"
    elif cropped_img[12, 112] == white and cropped_img[37, 37] != white: # Center of pixel at [0, 4]; white if letter "E"
        cv2.imshow("Cropped", cropped_img)
        return "E"
    elif cropped_img[62, 112] == white and cropped_img[12, 112] != white: # Center of pixel at [2, 4]; white if letter "D"
        cv2.imshow("Cropped", cropped_img)
        return "D"
    elif cropped_img[37, 112] == white and cropped_img[62, 122] != white:
        cv2.imshow("Cropped", cropped_img)
        return "B"                      # Define B as having [1, 4] and [2, 3] as white
    
    return None

fid_size = 53  # centimeters
# TODO fix calibration to setup cameraMatrix
cameraMatrix = np.array([[1.25649815e+03, 0.0, 7.12996774e+02],
                         [0.0, 1.25820533e+03, 4.69551858e+02], 
                         [0.0, 0.0, 1.0]], dtype=np.float32)
distCoeffs = np.array([[-3.72271817e-03, 5.33786890e-01, -4.99625728e-04, -1.65101232e-03,-1.78505927e+00]], dtype=np.float32)
# TODO testing program with hardcoded matrix since calibration session is not working
# TODO when game ready, dont forget to comment this out

def findTranslationAndRotation(image_pts):
    object_points = np.array([[-fid_size / 2.0, fid_size / 2.0, 0.0],
                              [fid_size / 2.0, fid_size / 2.0, 0.0],
                              [fid_size / 2.0, -fid_size / 2.0, 0.0],
                              [-fid_size / 2.0, -fid_size / 2.0, 0.0]], dtype=np.float32)
    #TODO: make calibration dynamic somehow
    _, rvec, tvec = cv2.solvePnP(object_points, np.array(image_pts, dtype=np.float32), 
                                 cameraMatrix, distCoeffs) #removed IPPE_SQUARE flag
    if rvec is not None:
        Rt = np.matrix(cv2.Rodrigues(rvec)[0])
        # TODO: remove all the yaw, pitch roll stuff at some point
        yaw, pitch, roll = rotToEul(Rt)
        R = Rt.T
        pose = -R * np.matrix(tvec)
        return tvec, (yaw, pitch, roll) #returning raw tvec rather than pose for testing.
    
    return "ERROR: TVEC nfound", "ERROR: rvec nfound"

# This function is kinda useless cuz we don't need it - TODO: remove it later
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

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    dim = 175
    p1 = np.array([
        [0, 0],
        [dim - 1, 0],
        [dim - 1, dim - 1],
        [0, dim - 1]], dtype="float32")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame, contoursR, contoursB = detect_target(frame)
        if len(contoursR):
            for cornersR in contoursR:
                c_rezR = order(cornersR[:, 0])
                hR, _ = cv2.findHomography(c_rezR, p1, cv2.RANSAC, 2)
                markerLetterR = None
                if hR is not None:
                    tagR = cv2.warpPerspective(frame, hR, (175, 175))
                    markerLetterR = determineLetter(cv2.cvtColor(tagR, cv2.COLOR_BGR2GRAY))
                    if markerLetterR:
                        tvecR, anglesR = findTranslationAndRotation(c_rezR)
                        print("Red Tag: " + markerLetterR)
                        print("CAMERA Position x=" + str(tvecR[2]) + " y=" + str(tvecR[0]) + " z=" + str(tvecR[1]))
                        print("Camera tilt -------   yaw=" + str(anglesR[0]) + " pitch=" + str(anglesR[1]) + " roll=" + str(anglesR[2]))

                        update_3d_plot(tvecR)

        if len(contoursB):
            for cornersB in contoursB:
                c_rezB = order(cornersB[:, 0])
                hB, _ = cv2.findHomography(c_rezB, p1, cv2.RANSAC, 2)
                markerLetterB = None
                if hB is not None:
                    tagB = cv2.warpPerspective(frame, hB, (175, 175))
                    markerLetterB = determineLetter(cv2.cvtColor(tagB, cv2.COLOR_BGR2GRAY))
                    if markerLetterB:
                        tvecB, anglesB = findTranslationAndRotation(c_rezB)
                        print("Blue Tag: " + markerLetterB)
                        print("CAMERA Position x=" + str(tvecB[2]) + " y=" + str(tvecB[0]) + " z=" + str(tvecB[1]))
                        print("Camera tilt -------   yaw=" + str(anglesB[0]) + " pitch=" + str(anglesB[1]) + " roll=" + str(anglesB[2]))
        
        cv2.imshow("Webcam Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
