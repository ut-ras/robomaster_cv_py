# UT RoboMaster CV - Stampede (Tag Detection Branch)

This branch builds on Eddie's tag detection python code. The goals are as follows:
1. Reliably and accurately detect only the DJI tags given a video feed
2. Reliably extract a translation vector and relative position from the detected DJI tag
3. Reliably obtain a field position and heading for the camera given the absolute tag positions.

## Important Notes (Aditya Pulipaka, adipu@utexas.edu, Discord: adipu24)

Currently, I have been trying to perfect this python code. It seems that it came from the official RoboMaster CV repository, but a version written in Python. While there are many components and subfolders within this repository that include armor plate detection and ranging, the main focus of this branch is the tag_detector subfolder and the calibration code. These are the important elements for tag-based localization.

**Function**
'newDetector.py' will detect multiple red and/or DJI tags at once and print their colors, letters, and relative positions.

**Calibration**

The camera matrix in tag_detector.py and newDetector.py is for my 2022 macbook air. If recalibration is desired, use 'basicCalib.py' with photos taken on your camera placed in the 'laptop calibration' directory instead of what's there currently. Change the *inner dimensions* of the chessboard at the beginning of 'basicCalib.py' if needed.
