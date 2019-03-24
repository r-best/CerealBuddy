import os
import cv2
import time
import signal
import logging
import picamera
import numpy as np
import RPi.GPIO as GPIO
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier

from processing import *


RESOLUTION = (640, 480)
PUMP_FORWARD = 18
PUMP_REVERSE = 17

def sig_handler(sig, frame):
    global running, x_train, y_train
    logging.info("Shutting down...")
    GPIO.cleanup()
    exit()


def pour(seconds):
    logging.info("Pouring")
    GPIO.output(PUMP_FORWARD, GPIO.HIGH)
    GPIO.output(PUMP_REVERSE, GPIO.LOW)
    time.sleep(seconds)
    GPIO.output(PUMP_FORWARD, GPIO.LOW)
    GPIO.output(PUMP_REVERSE, GPIO.LOW)
    logging.info("Done pouring")


def main():
    logging.basicConfig(format="%(levelname)s: %(message)s")
    logging.getLogger().setLevel(logging.INFO)

    logging.info("Initializing camera...")
    camera = picamera.PiCamera()
    camera.resolution = RESOLUTION
    camera.framerate = 24
    time.sleep(1)

    logging.info("Initializing pump...")
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(PUMP_FORWARD, GPIO.OUT)
    GPIO.setup(PUMP_REVERSE, GPIO.OUT)

    logging.info("Loading model...")
    train = np.genfromtxt('dataset.csv', delimiter=',')
    x_train = train[:,1:]
    y_train = np.uint8(train[:,0])
    clf = SGDClassifier(loss="log")
    clf.partial_fit(x_train, y_train, classes=[0,1])

    signal.signal(signal.SIGINT, sig_handler)

    logging.info("Starting inference loop")

    cerealCount = 0 # Increments for every consecutive time we see cereal, resetting if we lose it
    while True:
        logging.debug("Capturing image")
        image = np.empty((RESOLUTION[1], RESOLUTION[0], 3), dtype=np.uint8)
        camera.capture(image, 'bgr')

        logging.debug("Processing image")
        bellipse, gellipse, rellipse = getEllipse(image)
        features = extractFeatures(bellipse, gellipse, rellipse)

        # Predicted class is the index with the higher probability
        pred = np.argmax(clf.predict_proba([features])[0])

        if pred == 1:
            cerealCount = cerealCount + 1
            logging.info("Cereal detected! Count: {}".format(cerealCount))
        
        if cerealCount >= 10:
            cerealCount = 0
            pour(5)


if __name__ == "__main__":
    main()
