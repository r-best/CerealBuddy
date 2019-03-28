import picamera
import numpy as np
from sklearn.linear_model import SGDClassifier


def initCamera():
    """
    """
    camera = picamera.PiCamera()
    camera.resolution = RESOLUTION
    camera.framerate = 24
    time.sleep(1)
    return camera

def loadModel():
    """
    """
    train = np.genfromtxt('dataset.csv', delimiter=',')
    x_train = train[:,1:]
    y_train = np.uint8(train[:,0])
    clf = SGDClassifier(loss="log")
    clf.partial_fit(x_train, y_train, classes=[0,1])
    return x_train, y_train, clf
