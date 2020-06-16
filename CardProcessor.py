import cv2
import numpy as np

class CardProcessor:
    @staticmethod
    def findCards(image, toDraw = None):
        contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
            if len(approx) == 4 and peri > 1200:
                # compute the bounding box of the contour and use the
                # bounding box to compute the aspect ratio
                (x, y, w, h) = cv2.boundingRect(approx)
                box = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cropped = box[y: y + h, x: x + w]
                if toDraw is not None:
                    cv2.drawContours(toDraw, [cnt], 0,color=(255,0,0), thickness=5)
                return cropped
        return []