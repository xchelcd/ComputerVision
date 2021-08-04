import cv2
import os

path = 'model/'
modelPath = "{}rock-paper-scissors-model.h5".format(path)

if os.path.exists(modelPath):
    exec(open('game.py').read())
else:
    exec(open('train.py').read())
