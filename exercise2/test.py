from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('lena.jpg')
gray = cv2.imread("lena.jpg", 0)

print(type(img))
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()