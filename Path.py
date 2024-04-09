from sys import platform
if platform == "Windows":
    path = ""
    computer_cam_num = 1
    checkers_cam_num = 0
else:
    computer_cam_num = 2
    checkers_cam_num = 0
    path = "/Users/shelihendel/Documents/python/IP/DIP_Final_project/"