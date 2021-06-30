import cv2
import os

absolute_path = os.path.join(os.getcwd(), 'data', 'sample', '15_right.jpeg')
path = r'..\data\sample'

# load the image, convert it to grayscale, and blur it
image = cv2.imread(absolute_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (11, 11), 0)

# threshold the image to reveal light regions in the
# blurred image
thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)[1]

thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=4)

cv2.imshow('image', cv2.resize(image, (int(image.shape[1]/5), int(image.shape[0]/5))))
cv2.imshow('image', cv2.resize(thresh, (int(thresh.shape[1]/5), int(thresh.shape[0]/5))))
cv2.waitKey(0) 


