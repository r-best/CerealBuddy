import os
import cv2
import time
import signal
import logging
import numpy as np
import matplotlib.pyplot as plt

import cerealbuddy as cereal


x_train = []
y_train = []


def sig_handler(sig, frame):
    """
    """
    global running, x_train, y_train
    logging.info("Shutting down...")
    cv2.destroyAllWindows()
    logging.info("Saving dataset to file")
    train = np.insert(x_train, 0, y_train, axis=1)
    np.savetxt('dataset_temp.csv', train, delimiter=',', fmt='%s')
    os.remove('dataset.csv')
    os.rename('dataset_temp.csv', 'dataset.csv')

    logging.info("Done!")
    exit()

def main():
    global x_train, y_train
    logging.basicConfig(format="%(levelname)s: %(message)s")
    logging.getLogger().setLevel(logging.DEBUG)

    logging.info("Initializing camera...")
    camera = cereal.initCamera()

    logging.info("Loading model...")
    x_train, y_train, clf = cereal.loadModel()

    signal.signal(signal.SIGINT, sig_handler)

    logging.info("Starting inference loop")
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

        text = "This is cereal!" if pred == 1 else "This is not cereal"
        cv2.putText(image, text, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0,0,0), 3)
        cv2.imshow("Cereal Buddy", image)

        keycode = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if keycode == 121: # y key
            logging.debug("Prediction was right!")
            x_train = np.insert(x_train, len(x_train), features, axis=0)
            y_train = np.append(y_train, pred)
            clf.partial_fit([features], [pred])
        elif keycode == 110: # n key
            logging.debug("Prediction was wrong :(")
            correct = 0 if pred == 1 else 1
            x_train = np.insert(x_train, len(x_train), features, axis=0)
            y_train = np.append(y_train, correct)
            clf.partial_fit([features], [correct])
        else:
            logging.warn("The key pressed was not 'Y' or 'N', cannot add images to the dataset without knowing whether the prediction was correct!")


if __name__ == "__main__":
    main()
