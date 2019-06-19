#<<<<<<< HEAD
import cv2
import numpy as np

img = cv2.imread("mel.jpg")
img = cv2.resize(img, (1000, 1000), fx=1, fy=1)
#=======
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage import feature
import os

img = cv2.imread("test.jpg")
img = cv2.resize(img, (400, 400), fx=1, fy=1)
#>>>>>>> added pretrained CNN models. Added image preprocessing script.
height, width, channels = img.shape

newImg = np.zeros((height, width, 3), np.uint8)


imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(newImg, contours, -1, (0, 255, 0), 3)

#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
#cv2.erode(img, kernel, newImg)

#cv2.imshow("GRAY-SCALE", imgray)
#cv2.imshow("CONTOUR", newImg)
#=======
# grayscale, blur, threshold
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(imgray, (5, 5), 0)
thresh = cv2.threshold(blurred, 140, 170, cv2.THRESH_TOZERO_INV)[1]

# creating a mask for range extraction
mask = cv2.inRange(blurred, 0, 255)
masked = cv2.bitwise_and(blurred, 100, mask=mask)

contours, h = cv2.findContours(thresh, 1, 2)
cv2.drawContours(thresh, contours, -1, (255, 255, 255), 3)


#######################################
# IMAGE QUANTIZATION #

# pack the grayscale img into an array of float32
Z = imgray.reshape((-1, 1))
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 5
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape(imgray.shape)
####################################################


# display images
#cv2.imshow("GREY", imgray)
#cv2.imshow("ORIGINAL", img)
#cv2.imshow("THRESH", thresh)
#cv2.imshow("MASKED", masked)
#cv2.imshow("RES2", res2)


# setting up lists for iteration
images = [img, thresh, masked, res2]
names = ["ORIGINAL", "THRESH", "MASKED", "RES2"]
i = 1

# set matplot figure size
plt.rc('figure', figsize=[10, 10])

# print histograms
for img, name in zip(images, names):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.subplot(4, 1, i)
    i += 1
    plt.title(name)
    plt.plot(hist, color='k')
    plt.xlim([0, 256])
#plt.show()


'''
eps = 1e-7
neighbor = 2
radius = 1
lbp = feature.local_binary_pattern(imgray, neighbor, radius, method="var")
(hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, neighbor + 3), range=(0, neighbor + 2))

cv2.imshow("LBP", lbp)
cv2.imwrite(r'E:\PROGRAMS\GitHub\PROJEKT ZESPOLOWY\REPO\ProjektZespolowyKWKWKS\Preprocessed\lbptest3.jpg', lbp)
#####################################
'''
sourceOfImages = r'E:\PROGRAMS\GitHub\PROJEKT ZESPOLOWY\REPO\ProjektZespolowyKWKWKS\Samples\SickSamples'
destOfImages = r'E:\PROGRAMS\GitHub\PROJEKT ZESPOLOWY\REPO\ProjektZespolowyKWKWKS\Preprocessed'
images = []
names = []
eps = 1e-7
neighbor = 4
radius = 2
i = 0

for filename in os.listdir(sourceOfImages):
    img = cv2.imread(os.path.join(sourceOfImages, filename))
    if img is not None:
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lbp = feature.local_binary_pattern(imgray, neighbor, radius, method="var")
        cv2.imwrite(os.path.join(destOfImages, filename), lbp)
        #print(names)


#>>>>>>> added pretrained CNN models. Added image preprocessing script.
cv2.waitKey()
