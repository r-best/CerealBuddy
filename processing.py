import cv2
import numpy as np

def getHoughCircles(channel):
    return np.uint16(np.around(
        cv2.HoughCircles(channel, cv2.HOUGH_GRADIENT, 1, 60, param1=50, param2=55, minRadius=0, maxRadius=100)
    ))


def test(image):
    b, g, r = cv2.split(image)
    bcircles = getHoughCircles(b)
    gcircles = getHoughCircles(g)
    rcircles = getHoughCircles(r)
    for i in bcircles[0,:]:
        cv2.circle(image, (i[0],i[1]), i[2], (255,0,0), 2)
        cv2.circle(image, (i[0],i[1]), 2, (255,0,255), 3)
    for i in gcircles[0,:]:
        cv2.circle(image, (i[0],i[1]), i[2], (0,255,0), 2)
        cv2.circle(image, (i[0],i[1]), 2, (255,0,255), 3)
    for i in rcircles[0,:]:
        cv2.circle(image, (i[0],i[1]), i[2], (0,0,255), 2)
        cv2.circle(image, (i[0],i[1]), 2, (255,0,255), 3)
    
    cv2.imshow('detected circles',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
