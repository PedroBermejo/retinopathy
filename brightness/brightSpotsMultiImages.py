import cv2
import os

read_path = os.path.join(os.getcwd(), 'data', 'sample')

print(read_path)
for root, _, files in os.walk(read_path):
    for file_name in files:
        # load the image, convert it to grayscale, and blur it
        image_read_path = os.path.join(os.getcwd(), 'data', 'sample', file_name)
        image = cv2.imread(image_read_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)

        # threshold the image to reveal light regions in the
        # blurred image
        thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)[1]

        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=4)

        image_write_path = os.path.join(os.getcwd(), 'data', 'output', file_name)
        cv2.imwrite(image_write_path, thresh)




