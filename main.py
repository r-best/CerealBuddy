import cv2
import time
import signal
import logging
import picamera
import numpy as np
import matplotlib.pyplot as plt


RESOLUTION = (640, 480)
running = True


def sig_handler(sig, frame):
    global running
    logging.info("Shutting down...")
    running = False


def main():
    logging.basicConfig(format="%(levelname)s: %(message)s")

    logging.info("Initializing camera...")
    logging.getLogger().setLevel(logging.DEBUG)
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
    

if __name__ == "__main__":
    main()
