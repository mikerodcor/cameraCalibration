import numpy as np
import cv2 as cv

# Define chessboard size (number of inner corners in width and height)
chessboardSize = (7, 14)

# Define the size of the image used for calibration
frameSize = (1280, 720)

# Provided corner points in image coordinates
cnrs = np.array([
    [[491, 24]], [[529, 26]], [[573, 20]], [[616, 18]], [[659, 18]], [[704, 19]], [[746, 19]], [[486, 42]],
    [[529, 42]], [[570, 41]], [[616, 38]], [[659, 38]], [[708, 39]], [[751, 40]], [[476, 69]], [[521, 65]],
    [[568, 64]], [[615, 61]], [[662, 60]], [[712, 61]], [[759, 63]], [[468, 95]], [[516, 91]], [[564, 91]],
    [[613, 88]], [[663, 87]], [[717, 86]], [[764, 88]], [[458, 124]], [[509, 121]], [[560, 118]], [[612, 117]],
    [[665, 117]], [[722, 117]], [[773, 121]], [[448, 160]], [[501, 157]], [[556, 155]], [[611, 154]], [[667, 153]],
    [[727, 157]], [[782, 155]], [[438, 199]], [[491, 199]], [[550, 194]], [[611, 193]], [[670, 193]], [[734, 192]],
    [[791, 194]], [[426, 243]], [[485, 240]], [[547, 239]], [[609, 237]], [[673, 236]], [[740, 236]], [[801, 238]],
    [[416, 291]], [[478, 290]], [[545, 290]], [[608, 288]], [[675, 288]], [[746, 286]], [[811, 287]], [[405, 346]],
    [[466,348]],[[537, 347]], [[607, 345]], [[678, 345]], [[753, 344]], [[821, 343]], [[394, 408]], [[460, 409]], [[532, 409]],
    [[606, 409]], [[682, 408]], [[701, 407]], [[832, 404]], [[383, 479]], [[453, 482]], [[528, 484]], [[605, 484]],
    [[684, 486]], [[769, 481]], [[842, 476]], [[375, 550]], [[446, 555]], [[524, 559]], [[605, 561]], [[688, 560]],
    [[774, 555]], [[851, 548]],[[369, 623]], [[442, 631]], [[522, 638]], [[605, 640]], [[691, 638]], [[781, 632]], [[859, 623]]
], dtype=np.float32)

# Prepare object points based on chessboard size
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

# Convert object points and image points to the correct format
objPoints = [objp]  # List of object point arrays
imgPoints = [cnrs]  # List of image point arrays

# Perform camera calibration
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, frameSize, None, None)

# Check calibration success
if ret:
    print('Camera matrix:\n', cameraMatrix)
    print('\nDistortion parameters:\n', dist)
    print('\nRotation vectors:\n', rvecs)
    print('\nTranslation vectors:\n', tvecs)
else:
    print('Calibration failed')

######## Undistort
# Load an image for undistortion
img = cv.imread('WIN_20240826_19_31_22_Pro.jpg')
h, w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))

# Undistort the image
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
# Crop the image based on the ROI

cv.imwrite('distort_result.jpg', dst)

# Undistort with remapping
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (1280, 720), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
# Crop the image based on the ROI
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('distort_result_remap.jpg', dst)

# Compute and print reprojection error
mean_error = 0
for i in range(len(objPoints)):
    imgPoints2, _ = cv.projectPoints(objPoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgPoints[i], imgPoints2, cv.NORM_L2) / len(imgPoints2)
    mean_error += error

print("\nTotal error: {}".format(mean_error / len(objPoints)))
