import numpy as np
import cv2
import requests
from PreProcessor import PreProcessor
from CardProcessor import CardProcessor
set = int(input())
if set == 0:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 5)
cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
cv2.namedWindow('threshold',cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', 600,600)
cv2.resizeWindow('threshold', 600,600)
while(True):
    # Capture frame-by-frame
    if set==0:
        ret, frame = cap.read()
    else:
        url = "http://192.168.1.95:8080/shot.jpg"
        imgResp = requests.get(url)
        imgArr = np.array((bytearray(imgResp.content)), dtype=np.uint8)
        frame = cv2.imdecode(imgArr, -1)
    #copy the output
    output = frame
    thresh = PreProcessor.preProcessImage(frame)
    #draw only rectangle-alike countours wchich permieter is bigger than 'some value'
    cardImg = CardProcessor.findCards(thresh, toDraw= output)
    if len(cardImg) > 0:
        print(cardImg)
        cv2.imshow("Show Boxes", cardImg)


    # Display the resulting frame
    cv2.imshow('frame',output)
    cv2.imshow('threshold',thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
if set==0:
    cap.release()
cv2.destroyAllWindows()