import cv2
import numpy as np

def detect_red_target(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define red color range in HSV
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks to detect red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    
    # Apply mask to extract red regions
    red_regions = cv2.bitwise_and(frame, frame, mask=mask)
    gray = cv2.cvtColor(red_regions, cv2.COLOR_BGR2GRAY)
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
        corners = approx_corners.reshape(4, 2)
        return frame, corners, bounding_box_area
    
    return frame, None, 0

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame, corners, area = detect_red_target(frame)
        
        if corners is not None:
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
