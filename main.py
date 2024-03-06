import cv2
import os
import numpy as np
from image_sample import Camera_API


cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


