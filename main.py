import cv2
import os
import numpy as np
from image_sample import Camera_API


camera_api = Camera_API()

camera_api.open_camera()
frame = camera_api.read_frame()
camera_api.display_frame(frame)
camera_api.close_display_window()
camera_api.close_cameras()






