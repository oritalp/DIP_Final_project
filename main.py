import cv2
import pygame
import os
import numpy as np
from image_sample import Camera_API
import matplotlib.pyplot as plt
import board_utils
import checkers_utils
from Checkers import Checkers


pygame.init()
camera_api = Camera_API()
window_size = (640, 640)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Checkers")

checkers = Checkers(screen, camera_api)
checkers.main(window_size[0], window_size[1])






 