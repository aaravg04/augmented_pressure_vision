# from pred_util import *
import cv2
import numpy as np

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)

# look for chessboard in image and calibrate camera
def calibrate_camera(image) -> np.ndarray:
    # convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)

    if ret:
        corners = np.array([[corner for [corner] in corners]])
        c2 = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
        
        # testing purposes - draw chessboard to input image
        
        # cv2.drawChessboardCorners(image, (7, 7), c2, ret)
        # cv2.imshow('img', image)
        # cv2.waitKey(5000)
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp], c2, gray.shape[::-1], None, None)

        return (rvecs, tvecs, mtx, dist)

        # return P
    
    return -1   
    

print("calibrating image")

# get all cameras on device and take image from each

# projection matrics
ps = []

# cam images
all_camera_images = []

# set param -> n cameras
n = 1

# Get the list of all available camera indices
camera_indices = list(range(n))  # looking for n cameras

# Try to open each camera and take one image
for index in camera_indices:
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        c = -1
        while c == -1:
            ret, frame = cap.read()
            cv2.imshow(f"Camera {index} Capture", frame)
            cv2.waitKey(1)  # Display the frame for 1 ms
            if ret:
                # all_camera_images.append(frame)
                c = calibrate_camera(frame)
                if c != -1:
                    ps.append(c)
                    print(f"Image captured from camera {index}")

        cap.release()
    else:
        print(f"Camera {index} is not available")

for i, mat in enumerate(ps):
    if mat != -1:
        print(f"Camera {i} projection info:")
        print(mat)
    else:
        print(f"Camera {i} calibration failed - chessboard not found")

cv2.destroyAllWindows()
