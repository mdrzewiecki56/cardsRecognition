import cv2
import numpy as np

class PreProcessor:
    @staticmethod
    def preProcessImage(image):
        #convert to grayscale and apply some blur
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # get the image size
        img_w, img_h = np.shape(image)[:2]
        bkg_level = gray[int(img_h / 100)][int(img_w / 2)]
        thresh_level = bkg_level + 60
        # apply threshold
        _, thresh = cv2.threshold(gray, thresh_level, 255, cv2.THRESH_BINARY)
        return thresh