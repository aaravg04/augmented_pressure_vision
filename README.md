# Pressure Vision 3
Two more approaches to improve upon the current work done

1. take two camera inputs and use stereo vision to get the 3D pooling of the pressure inference

2. take two images and train a new model to estimate the pressure based on two image inputs instead of a single image 

## Approach Status

1: Stereo Vision 3D pooling implementation (`prediction/twoway-inf.py`)

- [x] camera projection calibration
- [x] calibration with multiple cameras and single chessboard
- [x] run inference on both camera input images post calibration
- [x] fix projection -> seems to be working, using panorama stitching for overlapping features to pool (use pix[i,j] = max(img1[i,j], img2[i,j]) instead of addition to avoid inflation of points where pressure estimated from both angles)
- [x] figure out how to pool inference in 3D (average, conv, something else?)
- [ ] run trials of accuracy against pressure sensor (single camera vs double camera approach to see which is better, mIoU metric because segmentation?)
- [ ] refactor code so it can be run easily by a user (right now its a very messy script)

2: Dual image pressure estimation model training

- [ ] TODO: plan once phase 1 completed (likely a very similar architecture to the preexisting model though, just for two image inputs instead of one)
