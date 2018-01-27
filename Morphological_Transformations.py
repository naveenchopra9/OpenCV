import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
import cv2

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
import cv2
#img2 = cv2.imread('/Users/naveen/Desktop/c1.jpeg' )
cap = cv2.VideoCapture(0)
while(1):
    
    # Take each frame
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    # Threshold the HSV image to get only blue colors
    kernel=np.ones((5,5),np.uint8)
    mask =cv2.inRange(hsv, lower_blue, upper_blue)
    erode=cv2.erode(mask,kernel,iterations=1)
    ditation=cv2.dilate(mask,kernel,iterations=1)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
    cv2.namedWindow('erode', cv2.WINDOW_NORMAL)
    cv2.namedWindow('ditation', cv2.WINDOW_NORMAL)
    cv2.namedWindow('opening', cv2.WINDOW_NORMAL)
    cv2.namedWindow('closing', cv2.WINDOW_NORMAL)
    cv2.namedWindow('gradient', cv2.WINDOW_NORMAL)
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('erode',erode)
    cv2.imshow('ditation',ditation)
    cv2.imshow('opening',opening)
    cv2.imshow('closing',closing)
    cv2.imshow('gradient',gradient)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
