# from pred_util import *
import cv2
import numpy as np
import torch
from prediction.pred_util import find_latest_checkpoint, get_hand_bbox, get_full_frame_1d, draw_bbox_full_frame, process_and_run_model, load_config
from recording.util import set_subframe, pressure_to_colormap
from pose.mediapipe_minimal import MediaPipeWrapper

# from demo_webcam import *

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)

chessboardCorners = []

# look for chessboard in image and calibrate camera
def calibrate_camera(image) -> np.ndarray:
    # convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)

    if ret:
        corners = np.array([[corner for [corner] in corners]])
        c2 = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
        chessboardCorners.append(corners)
        # print(corners)
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
n = 2
start_idx = 1 # use 0 if starting with integrated camera on laptop

# Get the list of all available camera indices
camera_indices = list(range(n))  # looking for n cameras

# Try to open each camera and take one image
for index in camera_indices:
    cap = cv2.VideoCapture(index + start_idx)
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
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print(f"Camera {index} is not available")

for i, mat in enumerate(ps):
    if mat != -1:
        print(f"Camera {i} projection info:")
        print(mat)
    else:
        print(f"Camera {i} calibration failed - chessboard not found")

cv2.destroyAllWindows()

# actual pressure processing
disp_x = 640
disp_y = 480

captures = [cv2.VideoCapture(i + start_idx) for i in range(n)]
mp_wrapper = MediaPipeWrapper()
disp_frame = np.zeros((disp_y, disp_x, 3), dtype=np.uint8)
config = load_config("paper")

H, mask = cv2.findHomography(chessboardCorners[0], chessboardCorners[1], cv2.RANSAC, 5.0)
print(f"projection matrix {H} computed")

if torch.cuda.is_available():
    best_model = torch.load(find_latest_checkpoint("paper"))
else:
    best_model = torch.load(find_latest_checkpoint("paper"), map_location=torch.device('cpu'))


try:
    while True:
        frames = []
        for i, cap in enumerate(captures):
            ret, camera_frame = cap.read()
            if ret:
                # frames.append(camera_frame)
                # cv2.imshow(f'Camera {captures.index(cap)}', camera_frame)
                base_img = camera_frame

                bbox = get_hand_bbox(base_img, mp_wrapper)
                crop_frame = base_img[bbox['min_y']:bbox['max_y'], bbox['min_x']:bbox['max_x'], ...]
                crop_frame = cv2.resize(crop_frame, (448, 448))    # image is YX

                force_pred = process_and_run_model(crop_frame, best_model, config)
                force_pred_full_frame = get_full_frame_1d(force_pred, bbox)
                force_pred_color_full_frame = pressure_to_colormap(force_pred_full_frame)

                # added for crappier cameras (i.e. my laptop camera lol)
                force_pred_color_full_frame = cv2.resize(force_pred_color_full_frame, (base_img.shape[1], base_img.shape[0]))
                # print(base_img.shape)
                # print(force_pred_color_full_frame.shape)

                overlay_frame = cv2.addWeighted(base_img, 0.6, force_pred_color_full_frame, 1.0, 0.0)
                draw_bbox_full_frame(overlay_frame, bbox)

                set_subframe(0, base_img, disp_frame, title='Raw Camera Frame')
                set_subframe(1, crop_frame, disp_frame, title='Network Input')
                set_subframe(2, overlay_frame, disp_frame, title='Network Output with Overlay')
                set_subframe(3, pressure_to_colormap(force_pred_full_frame), disp_frame, title='Network Output')

                # force_pred_full_frame is 1080,1920
                # scaled + broadcasted to 3d by p2c

                # p2c(force_pred_full_frame) = 1080,1920,3 
                # print(force_pred_full_frame.shape)
                # cv2.imshow("im not insane right", force_pred_full_frame)
                frames.append(force_pred_full_frame)
                # print(pressure_to_colormap(force_pred_full_frame).shape)
                cv2.imshow(f"cam {i} estimation", disp_frame)
            
            else:
                frames.append(None)
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # we have inference from all frames (or lack thereof -> None)
        # project each one togehter space and pool
        final_dims = (disp_y, disp_x)
        # summed = np.zeros((disp_y, disp_x))        
        summed = np.zeros((1080,1920))
        """
        TODO: 
        lock to 2 camera input
        
        compute homography from second image w.r.t first 
        use chessbaord image points for homography ->  M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)


        stitch by similar points to pano -> 
            
            out = cv.warpPerspective(src2,np.linalg.inv(homography),(b+n, a),flags=cv.INTER_LINEAR)
            ei2 = out
            ei2[:a,:b] = src1
        
        """
        src1, src2 = frames
        a,b = src1.shape
        m,n2 = src2.shape
        out = cv2.warpPerspective(src2,H,(b+n2, a),flags=cv2.INTER_LINEAR)
        ei2 = out
        # ei2[:a,:b] += src1

        # take max at each point
        ei2[:a, :b] = np.maximum(ei2[:a, :b], src1)

        # summed = cv2.resize(summed, final_dims)
        # print(summed)
        
        # set_subframe(3, pressure_to_colormap(summed), disp_frame, title='pooled')

        cv2.imshow(f"pooled from {n} cams", pressure_to_colormap(ei2))

        # cv2.imshow(f"cam {i} estimation", disp_frame)


finally:
    for cap in captures:
        cap.release()

    cv2.destroyAllWindows()

