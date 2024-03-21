import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from image_sample import Camera_API

ref_img = cv2.imread("images_taken/new_alligned.jpg")
 
camera_api = Camera_API()
camera_api.close_cameras()
camera_api.open_camera()
camera_api.stream_video(ref_img,"z", save_frame="video_exam", verbose=True)






