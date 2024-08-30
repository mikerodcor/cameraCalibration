import numpy as np
import cv2 as cv
from numpy.matrixlib.defmatrix import matrix


def perpectiveTransform( pts1, pts2, img):
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    result = cv.warpPerspective(img, matrix, (820, 1230))
    return result


# Define chessboard size (number of inner corners in width and height)
chessboardSize = (7, 14)

# Define the size of the image used for calibration
frameSize = (1280, 720)

# Provided corner points in image coordinates
cnrs = np.array([
    [[488, 58]],  [[525, 54]],  [[564, 51]],  [[605,49]],   [[644,47]],   [[687, 47]],  [[727, 46]],
    [[481, 79]],  [[521,75]],   [[561,71]],   [[604, 69]],  [[645, 67]],  [[690,65]],   [[732, 66]],
    [[473, 101]], [[516, 98]],  [[559,96]],   [[603, 92]],  [[647, 91]],  [[694, 89]],  [[737, 89]],
    [[468,128]],  [[511, 124]], [[557, 120]], [[601, 117]], [[648, 116]], [[698, 114]], [[743, 114]],
    [[459, 156]], [[506, 153]], [[553,149]],  [[602, 146]], [[650,153]],  [[702, 142]], [[750,142]],
    [[451, 190]], [[499, 187]], [[550, 183]], [[600, 180]], [[652, 178]], [[707, 176]], [[758, 176]],
    [[442, 226]], [[493, 223]], [[545, 220]], [[599, 217]], [[654, 215]], [[713, 214]], [[766, 212]],
    [[434, 266]], [[487, 262]], [[542, 260]], [[599, 258]], [[657, 256]], [[718, 254]], [[774, 252]],
    [[426, 312]], [[481, 309]], [[539, 306]], [[598, 304]], [[659, 302]], [[724, 300]], [[783, 298]],
    [[417, 361]], [[464, 360]], [[535, 358]], [[598, 356]], [[662, 354]], [[730, 351]], [[793, 348]],
    [[408, 416]], [[468, 415]], [[532, 414]], [[598, 412]], [[665, 410]], [[737,407]],  [[802, 404]],
    [[399, 473]], [[462, 480]], [[528, 480]], [[598, 479]], [[669, 477]], [[744, 474]], [[812, 468]],
    [[393, 541]], [[458, 545]], [[526, 546]], [[598, 546]], [[672, 544]], [[750,539]],  [[820, 533]],
    [[388, 607]], [[454, 613]], [[524, 615]], [[598, 616]], [[676,616]],  [[755, 614]], [[829,602]]
], dtype=np.float32)



# Prepare object points based on chessboard size
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

# Convert object points and image points to the correct format
objPoints = [objp]  # List of object point arrays
imgPoints = [cnrs]  # List of image point arrays
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
img = cv.imread('imgTest/WIN_20240828_17_47_33_Pro.jpg')
# Load an image for undistortion
h, w = img.shape[:2]

# Undistort the image
dst = cv.undistort(img, cameraMatrix, dist, None, None)


#cv.imread('distort_result.jpg', dst)

# Undistort with remapping
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, None, (1480, 920), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

#Resize image distorted and transformed
resizedst = cv.resize(dst, (1280, 720))

pt1 = np.float32([[423,35], [794, 38], [325, 594], [906, 594]])
pt2 = np.float32([[0, 0], [582, 0],[0, 873],[465, 873]])
ptfm = perpectiveTransform(pt1, pt2, resizedst)

cv.imwrite('distort_result_remap.jpg', ptfm)

# Compute and print reprojection error
mean_error = 0
for i in range(len(objPoints)):
    imgPoints2, _ = cv.projectPoints(objPoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgPoints[i], imgPoints2, cv.NORM_L2) / len(imgPoints2)
    mean_error += error

print("\nTotal error: {}".format(mean_error / len(objPoints)))