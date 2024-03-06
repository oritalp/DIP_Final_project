import cv2
import os
import numpy as np


cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("ori is in the house")

print("sheli")

print("this is only for the new branch")

