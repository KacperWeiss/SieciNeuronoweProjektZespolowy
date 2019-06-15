import cv2
import numpy as np

img = cv2.imread("mel.jpg")
img = cv2.resize(img, (1000, 1000), fx=1, fy=1)
height, width, channels = img.shape

newImg = np.zeros((height, width, 3), np.uint8)

imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(newImg, contours, -1, (0, 255, 0), 3)

#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
#cv2.erode(img, kernel, newImg)

cv2.imshow("GRAY-SCALE", imgray)
cv2.imshow("CONTOUR", newImg)
cv2.waitKey()
