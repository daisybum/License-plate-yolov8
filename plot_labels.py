import numpy as np
import cv2
from PIL import Image
from ultralytics.utils.plotting import Colors, Annotator

labels = np.genfromtxt('data\\102_1_20200921-070000-000_225.txt', delimiter=" ")
img = cv2.imread('data\\102_1_20200921-070000-000_225.jpg', cv2.IMREAD_COLOR)

colors = Colors()
ann = Annotator(img)

for label in labels:
    cls = int(label[0])
    box = list(np.rint(label[1:]))
    if cls == 0:
        label = 'License plate'
        color = colors.palette[0]
    elif cls == 2:
        label = 'car'
        color = colors.palette[1]
    elif cls == 7:
        label = 'truck'
        color = colors.palette[2]

    ann.box_label(box, label=label, color=color)

cv2.imshow('result', ann.result())
cv2.waitKey(0)
cv2.destroyAllWindows()
