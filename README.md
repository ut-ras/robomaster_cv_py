# UT RoboMaster CV - Stampede (Tag Detection Branch)

This branch builds on Eddie's tag detection python code. The goals are as follows:
1. Reliably and accurately detect only the DJI tags given a video feed
2. Reliably extract a translation vector and relative position from the detected DJI tag
3. Reliably obtain a field position and heading for the camera given the absolute tag positions.

## Important Notes

Currently, I (Aditya Pulipaka) have been trying to perfect this python code. It seems that it came from the official RoboMaster CV repository, but a version written in Python. While there are many components and subfolders within this repository that include armor plate detection and ranging, the main focus of this branch is the tag_detector subfolder and the calibration code. These are the important elements for tag-based localization.

**Solved Issues**:
- White border around tag would often be recognized along with tag. Incorporated another check into the 'determineLetter()' function to check for a white space at '[5,5]', which eliminated the error.
- Small red square inside tag A would often be detected. This was avoided by increasing the minimum area threshold for contour generation.
    - Another way to solve this would be by checking for the presence of white color inside the detected contour, but as of now, this alternate strategy has not been necessary.
- Other square or rectangular shapes in frame would be detected. This behavior has currently been minimized by returning 'false' from the 'determineColor()' function if neither blue nor red is detected and not continuing with the contour if 'false' is returned from 'determineColor()'.

