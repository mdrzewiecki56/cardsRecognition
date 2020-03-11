import numpy as np
import cv2
import time
import requests
url = "http://192.168.1.95:8080/shot.jpg"
#cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FPS, 5)
cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
cv2.namedWindow('threshold',cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', 600,600)
cv2.resizeWindow('threshold', 600,600)
while(True):
    imgResp = requests.get(url)
    imgArr = np.array((bytearray(imgResp.content)), dtype=np.uint8)
    frame = cv2.imdecode(imgArr, -1)
    # Capture frame-by-frame
    #ret, frame = cap.read()
    #frame = cv2.imread("image.png")
    output = frame
    # Convert to grayscale and blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    #get the image size
    img_w, img_h = np.shape(frame)[:2]
    bkg_level = gray[int(img_h / 100)][int(img_w / 2)]
    thresh_level =  bkg_level + 60
    #apply threshold
    _, thresh = cv2.threshold(gray, thresh_level, 255, cv2.THRESH_BINARY)
    #find countours (all)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    #draw only rectangle-alike countours wchich permieter is bigger than 'some value'
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and peri > 1000:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            cv2.drawContours(output, [cnt], 0,color=(255,0,0), thickness=5)
            print(peri)



    # Display the resulting frame
    cv2.imshow('frame',output)
    cv2.imshow('threshold',thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()