import cv2
import os
import numpy as np

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

        thresh = np.where(thresh != 0, 255, 0)
        
        thresh = thresh.astype(np.uint8)

        # guardar 800 + imagenes, guardar original y blanco y negro
        # investigar sobre segmentacion con imagenes
        # convertir a pillow y guardar con pillow como tif
        # estudiar autoencoder
        # tercer imagen de git guardar con 0, 1, 2 y en tif
        image_write_path = os.path.join(os.getcwd(), 'data', 'output', file_name)
        cv2.imwrite(image_write_path, thresh)




