import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
import cv2

img = cv2.imread('/Users/naveen/Desktop/project2.jpg',cv2.IMREAD_COLOR  )

px = img[10,10]

blue = img[100,100,0]

print("naveen",px,blue)

b,g,r=cv2.split(img)

img2 = cv2.merge([r,g,b])

plt.subplot(121)
plt.imshow(img) # expects distorted color
plt.subplot(122)
plt.imshow(img2) # expect true color
plt.show()

cv2.imshow('bgr image',img) # expects true color
cv2.imshow('rgb image',img2) # expects distorted color
cv2.waitKey(0)
cv2.destroyAllWindows()

#cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#cv2.imshow('image',img)
#cv2.imwrite('messigray.png',img)
#k=cv2.waitKey(0)
##cv2.destroyAllWindows()
#if k == 27:         # wait for ESC key to exit
#    cv2.destroyAllWindows()
#elisf k == ord('s'): # wait for 's' key to save and exit
#    cv2.imwrite('messigray.png',img)
#    cv2.destroyAllWindows()
