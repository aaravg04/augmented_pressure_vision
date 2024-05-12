# Pressure Vision 3
Two more approaches to improve upon the current work done

1. take two camera inputs and use stereo vision to get the 3D pooling of the pressure inference

2. take two images and train a new model to estimate the pressure based on two image inputs instead of a single image 

## Approach Status

1: Stereo Vision 3D pooling implementation (`prediction/twoway-inf.py`)

- [x] camera projection calibration
- [x] calibration with multiple cameras and single chessboard
- [x] run inference on both camera input images post calibration
- [ ] fix projection -> pooling isn't being done properly, distortion/resize issue?
- [x] figure out how to pool inference in 3D (average, conv, something else?)
- [ ] run trials of accuracy against pressure sensor (buy 2 cameras + pressure sensor from provided resources in paper)
- [ ] refactor code so it can be run easily by a user (right now its a very messy script)

2: Dual image pressure estimation model training

- [ ] TODO: plan once phase 1 completed
