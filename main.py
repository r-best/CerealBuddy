import cv2
import time
import signal
import logging
import picamera
import numpy as np
import matplotlib.pyplot as plt

from processing import *


RESOLUTION = (640, 480)
running = True


def sig_handler(sig, frame):
    global running
    logging.info("Shutting down...")
    running = False


def main():
    logging.basicConfig(format="%(levelname)s: %(message)s")
    logging.getLogger().setLevel(logging.DEBUG)

    logging.info("Initializing camera...")
    camera = picamera.PiCamera()
    camera.resolution = RESOLUTION
    camera.framerate = 24
    time.sleep(1)

    signal.signal(signal.SIGINT, sig_handler)

    logging.info("Starting inference loop")
    while running is True:
        logging.debug("Capturing image")
        image = np.empty((RESOLUTION[1], RESOLUTION[0], 3), dtype=np.uint8)
        camera.capture(image, 'bgr')

        logging.debug("Processing image")
        bellipse, gellipse, rellipse = getEllipse(image)
        features = extractFeatures(bellipse, gellipse, rellipse)

        cv2.imshow("", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
