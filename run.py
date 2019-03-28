import os
import cv2
import time
import signal
import logging
import numpy as np
import RPi.GPIO as GPIO
import matplotlib.pyplot as plt

import cerealbuddy as cereal


SERVO = 18
PUMP_FORWARD = 27
PUMP_REVERSE = 17

pwm = None


def sig_handler(sig, frame):
    """
    """
    global pwm
    logging.info("Shutting down...")
    pwm.stop()
    GPIO.cleanup()
    exit()

def pour(seconds):
    """
    """
    logging.info("Pouring")
    GPIO.output(PUMP_FORWARD, GPIO.LOW)
    GPIO.output(PUMP_REVERSE, GPIO.HIGH)
    time.sleep(seconds)
    GPIO.output(PUMP_FORWARD, GPIO.LOW)
    GPIO.output(PUMP_REVERSE, GPIO.LOW)

def target(x):
    """
    """
    global pwm
    logging.info("Targeting")
    pwm.start(2.5)
    target = 9 - ((x / RESOLUTION[0]) * 4 + 5) + 5
    print(target)
    pwm.ChangeDutyCycle(target)

def main():
    global pwm
    logging.basicConfig(format="%(levelname)s: %(message)s")
    logging.getLogger().setLevel(logging.INFO)

    logging.info("Initializing camera...")
    camera = cereal.initCamera()

    logging.info("Loading model...")
    _, _, clf = cereal.loadModel()

    logging.info("Initializing motor...")
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(SERVO, GPIO.OUT)
    pwm = GPIO.PWM(SERVO, 50)

    logging.info("Initializing pump...")
    GPIO.setup(PUMP_FORWARD, GPIO.OUT)
    GPIO.setup(PUMP_REVERSE, GPIO.OUT)

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

        if features[0] < 5000 or features[1] < 5000 or features[2] < 5000:
            pred = 0
        else:
            # Predicted class is the index with the higher probability
            pred = np.argmax(clf.predict_proba([features])[0])

        if pred == 1:
            cerealCount = cerealCount + 1
            logging.info("Cereal detected! Count: {}".format(cerealCount))
        else:
            if cerealCount > 0: logging.info("Cereal lost, resetting counter")
            cerealCount = 0
        
        if cerealCount >= 10:
            cerealCount = 0
            target(np.mean((bellipse[0][0], gellipse[0][0], rellipse[0][0])))
            pour(20)


if __name__ == "__main__":
    main()
