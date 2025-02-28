import cv2
import numpy as np

def detect_red_target(frame, color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define red color range in HSV
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    lowerBlue = np.array([100,80,160])
    upperBlue = np.array([120,255,255])
    
    # Create masks to detect red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    maskRed = mask1 + mask2

    maskBlue = cv2.inRange(hsv, lowerBlue, upperBlue)
    
    # Apply mask to extract red regions
    red_regions = cv2.bitwise_and(frame, frame, mask=maskRed)
    blue_regions = cv2.bitwise_and(frame, frame, mask = maskBlue)

    gray = None
    if color:
        gray = cv2.cvtColor(red_regions, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(blue_regions, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    approx_corners = None
    bounding_box_area = 0
    min_area_threshold = 500  # Ignore small detections
    
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area < min_area_threshold:
                continue  # Ignore small contours
            
            # Convexity check
            if not cv2.isContourConvex(approx):
                continue
            
            approx_corners = approx
            bounding_box_area = area
            break
    
    if approx_corners is not None:
        cv2.drawContours(frame, [approx_corners], -1, (0, 255, 0), 3)
        corners = approx_corners.reshape(4, 1, 2)
        return frame, corners, bounding_box_area
    
    return frame, np.array([]), 0

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
    cv2.imshow("Cropped", cropped_img)

    # [y, x]: [0, 0] = top left pixel
    # Each pixel seems to be like 25 
    if cropped_img[12, 37] != white and cropped_img[12, 62] == white:    # Center of pixel at [0, 1]; not white if letter "A"
        return "A"
    elif cropped_img[12, 12] != white and cropped_img[12, 37] == white:  # Center of pixel at [0, 0]; not white if letter "C"
        return "C"
    elif cropped_img[12, 112] == white: # Center of pixel at [0, 4]; white if letter "E"
        return "E"
    elif cropped_img[62, 112] == white: # Center of pixel at [2, 4]; white if letter "D"
        return "D"
    elif cropped_img[37, 112] == white and cropped_img[62, 87] == white:
        return "B"                      # Must be "B" if none of the above
    
    return None

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
    #TODO: make calibration dynamic somehow
    _, rvec, tvec = cv2.solvePnP(object_points, np.array(h, dtype=np.float32), cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)

    if rvec is not None:
        Rt = np.matrix(cv2.Rodrigues(rvec)[0])
        # TODO: remove all the yaw, pitch roll stuff at some point
        yaw, pitch, roll = rotToEul(Rt)
        R = Rt.T
        pose = -R * np.matrix(tvec)

        print("CAMERA Position x=" + str(pose[2]) + " y=" + str(pose[0]) + " z=" + str(pose[1]))
        print("Camera tilt -------   yaw=" + str(yaw) + " pitch=" + str(pitch) + " roll=" + str(roll))
        return pose, (yaw, pitch, roll)
    
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
        
        frame, corners, area = detect_red_target(frame, 1)
        h = None
        markerLetter = None
        if corners.size>0:
            x, y, w, h = cv2.boundingRect(corners)
            c_rez = order(corners[:, 0])
            h, _ = cv2.findHomography(c_rez, p1, cv2.RANSAC, 2)

        if h is not None:
            tag = cv2.warpPerspective(frame, h, (175, 175))
            markerLetter = determineLetter(cv2.cvtColor(tag, cv2.COLOR_BGR2GRAY))
            if markerLetter:
                tvec, angles = findTranslationAndRotation(c_rez)
                print("Tag: " + markerLetter)


        
        if corners is not None and markerLetter:
            print(f"Corners: {corners.tolist()} | Area: {area}")
        else:
            print("No valid red tag detected.")
        
        cv2.imshow("Webcam Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
