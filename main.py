import cv2
import os
import numpy as np
from image_sample import Camera_API
import matplotlib.pyplot as plt
import board_utils
import checkers_utils

#camera_api = Camera_API()

#camera_api.open_camera()
#frame = camera_api.read_frame()
#camera_api.display_frame(frame)
#camera_api.close_display_window()
#camera_api.close_cameras()
board_bin = [[0 for _ in range(8)] for _ in range(8)]
board_bin[0][0] = 1
board_bin[1][1] = 1
bord_color = [["" for _ in range(8)] for _ in range(8)]
bord_color[0][0] = "bp"
bord_color[1][1] = "rp"
old_board = [board_bin, bord_color]

img_to_al = plt.imread("images_taken/img_6.jpg")
ref_img = plt.imread("images_taken/ref_img.jpg")

verbose = True
max_lines = 14
crop_width = 70
crop_height = 70

aligned_img,_ = board_utils.align_board(img_to_al, ref_img, verbose=False)
intersect = board_utils.get_intersections(aligned_img, verbose=True)
pawns_location = board_utils.pawns_location()
new_board = checkers_utils.ip_to_matrix(intersect, pawns_location)
pos = checkers_utils.matrix_to_move(new_board, old_board)
print("done")



