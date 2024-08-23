
import numpy as np
import cv2 as cv
import glob

chessboardSize= (13,9)
frameSize = (1280,720)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((chessboardSize[0] * chessboardSize[1],3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1,2)

objPoints = []
imgPoints = []

images = glob.glob('*.jpg')

for image in images:
    print(image)
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    if ret == True:
        print("No")
        objPoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgPoints.append(corners)

        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)

cv.destroyAllWindows()


#CALIBRATION

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, frameSize, None, None)

print('Camera calibrated: ', ret)
print('\nCamera matrix: ', cameraMatrix)
print('\nDistortion parameters: ', dist)
print('\nRotation vectors: ', rvecs)
print('\nTranslation vectors: ', tvecs)


########Undistort
img = cv.imread('0.3_8.jpg')
h, w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist (h,w), 1, (h,w))

#undistort
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
#crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('distort_result.jpg', dst)

#undistort with remapping
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (h,w), 5)
#crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('distort_result.jpg', dst)

#Reprojection error
mean_error = 0

for i in range(len(objPoints)):
    imgPoints2, _ = cv.projectPoints(objPoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgPoints[i], imgPoints2, cv.NORM_L2)/len(imgPoints2)
    mean_error+= error

    print("\nTotal error: {}".format(mean_error/len(objPoints)))
    print('\n\n\n')