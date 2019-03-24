import cv2
import math
import numpy as np


def extractFeatures(b, g, r):
    """
    """
    b_center = np.array((b[0][0], b[0][1]))
    g_center = np.array((g[0][0], g[0][1]))
    r_center = np.array((r[0][0], r[0][1]))

    b_area = math.pi * b[1][0] * b[1][1]
    g_area = math.pi * g[1][0] * g[1][1]
    r_area = math.pi * r[1][0] * r[1][1]

    # Area of the triangle formed by the ellipses' centerpoints, measures how
    # close together (concentric) the circles are
    dist_similarity = (np.linalg.norm(b_center - g_center) * np.linalg.norm(b_center - r_center)) / 2

    # The standard deviation of the circles' areas, measures how similar in size they are
    size_similarity = np.std([b_area, g_area, r_area])

    logging.debug("Blue Area: {}".format(b_area))
    logging.debug("Green Area: {}".format(g_area))
    logging.debug("Red Area: {}".format(r_area))
    logging.debug("Distance Similarity: {}".format(dist_similarity))
    logging.debug("Size Similarity: {}".format(size_similarity))

    return np.array([b_area, g_area, r_area, dist_similarity, size_similarity])


def getEllipse(image):
    """
    """
    getEdges = lambda channel: np.uint8(np.around(
        cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
    ))

    b, g, r = [cv2.threshold(getEdges(x), 55, 255, cv2.THRESH_BINARY)[1] for x in cv2.split(image)]

    b, bcontours, hierarchy = cv2.findContours(b, cv2.RETR_TREE, 1)
    bellipse = cv2.fitEllipse(bcontours[np.argmax([cv2.contourArea(x) for x in bcontours])])
    # image = cv2.drawContours(image, bcontours, -1, (255,0,0), 3)

    g, gcontours, hierarchy = cv2.findContours(g, cv2.RETR_TREE, 1)
    gellipse = cv2.fitEllipse(gcontours[np.argmax([cv2.contourArea(x) for x in gcontours])])
    # image = cv2.drawContours(image, gcontours, -1, (0,255,0), 3)

    r, rcontours, hierarchy = cv2.findContours(r, cv2.RETR_TREE, 1)
    rellipse = cv2.fitEllipse(rcontours[np.argmax([cv2.contourArea(x) for x in rcontours])])
    # image = cv2.drawContours(image, rcontours, -1, (0,0,255), 3)

    cv2.ellipse(image, bellipse, (255, 0, 0), 3)
    cv2.circle(image, (int(bellipse[0][0]), int(bellipse[0][1])), 2, (255, 0, 0))
    cv2.ellipse(image, gellipse, (0, 255, 0), 3)
    cv2.circle(image, (int(gellipse[0][0]), int(gellipse[0][1])), 2, (0, 255, 0))
    cv2.ellipse(image, rellipse, (0, 0, 255), 3)
    cv2.circle(image, (int(rellipse[0][0]), int(rellipse[0][1])), 2, (0, 0, 255))
    
    # Each ellipse object contains: (x, y), (Ma, ma), angle
    return bellipse, gellipse, rellipse
