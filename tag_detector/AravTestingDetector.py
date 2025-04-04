import cv2
import numpy as np

TAG_SIZE = 5.0  # cm

def get_camera_matrix(cap):
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    focal_length = 980  # estimate
    cx = width / 2
    cy = height / 2
    camera_matrix = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))
    return camera_matrix, dist_coeffs

def get_object_points(tag_size):
    half = tag_size / 2
    return np.array([
        [-half, -half, 0],
        [ half, -half, 0],
        [ half,  half, 0],
        [-half,  half, 0]
    ], dtype=np.float32)

def detect_red_target(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    red_regions = cv2.bitwise_and(frame, frame, mask=mask)
    gray = cv2.cvtColor(red_regions, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            if cv2.contourArea(approx) > 500:
                return approx.reshape(4, 2).astype(np.float32)
    return None

def draw_axes(frame, rvec, tvec, camera_matrix, dist_coeffs, corner):
    axis = np.float32([[0,0,0], [2,0,0], [0,2,0], [0,0,-2]])
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
    origin = tuple(corner.astype(int))
    for i in range(1, 4):
        pt = tuple(imgpts[i].ravel().astype(int))
        cv2.line(frame, origin, pt, (0,0,255), 2)

def invert_pose(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    R_inv = R.T
    tvec_inv = -R_inv @ tvec
    rvec_inv, _ = cv2.Rodrigues(R_inv)
    return rvec_inv, tvec_inv

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    camera_matrix, dist_coeffs = get_camera_matrix(cap)
    object_points = get_object_points(TAG_SIZE)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        corners = detect_red_target(frame)
        if corners is not None:
            cv2.polylines(frame, [corners.astype(int)], True, (0, 255, 0), 2)
            success, rvec, tvec = cv2.solvePnP(object_points, corners, camera_matrix, dist_coeffs)

            if success:
                # Position of tag relative to camera
                tx, ty, tz = tvec.flatten()
                print(f"[Tag in camera frame] x={tx:.2f}, y={ty:.2f}, z={tz:.2f} cm")

                # Position of camera relative to tag (for localization)
                rvec_inv, tvec_inv = invert_pose(rvec, tvec)
                cx, cy, cz = tvec_inv.flatten()
                print(f"[Camera in tag frame] x={cx:.2f}, y={cy:.2f}, z={cz:.2f} cm\n")

                draw_axes(frame, rvec, tvec, camera_matrix, dist_coeffs, corners[0])

        cv2.imshow("Tag Localization", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
