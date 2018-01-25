import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
import cv2
#from skimage.io import imread

img2 = cv2.imread('/Users/naveen/Desktop/c1.jpeg' )
img1 = cv2.imread('/Users/naveen/Desktop/c2.jpeg' )
# I want to put IMGAGE 2 on top-left corner, So I create a ROI
rows,cols,channels= img2.shape

roi = img1[0:rows, 0:cols ]

# Now create a mask of image 2 and create its inverse mask also
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)

mask_inv = cv2.bitwise_not(mask)
# Now black-out the area of logo in ROI

img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
# Take only region of logo from logo image.

img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

# Put logo in ROI and modify the main image
dst = cv2.add(img1_bg,img2_fg)

img1[0:rows, 0:cols ] = dst

cv2.imshow('res',img1)

cv2.waitKey(0)
cv2.destroyAllWindows()

