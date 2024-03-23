import cv2
import pygame
import checkers_utils
from Checkers import Checkers
from tkinter import *
from tkinter import ttk


#Ori's check
ref_img = cv2.imread("images_taken/new_alligned.jpg")
 
#Shelly's check
computer_cam = cv2.VideoCapture(2)
# GUI start
root= Tk()
root.protocol("WM_DELETE_WINDOW", checkers_utils.on_closing)
root.geometry("750x250")
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
root.rowconfigure(1, weight=1)
root.rowconfigure(2, weight=1)
main_label = Label(root, text="Welcome to Interactive Checkers!", font=('Helvetica 17 bold'))
main_label.grid(row = 0, pady = 2)
buttons_frame = Frame(root)
buttons_frame.grid(row = 1, pady = 2)
red_frame = Frame(buttons_frame)
red_frame.pack(side=LEFT, expand=True, padx = 30)
red_label = Label(red_frame, text="Choose a picture for the red player:", font=('Helvetica 14'))
red_label.pack()
red_button = ttk.Button(red_frame, text="Choose Picture", command=lambda : checkers_utils.choose_red(computer_cam))
red_button.pack()
white_frame = Frame(buttons_frame)
white_frame.pack(side=RIGHT, expand=True, padx = 30)
white_label = Label(white_frame, text="Choose a picture for the white player:", font=('Helvetica 14'))
white_label.pack()
white_button = ttk.Button(white_frame, text="Choose Picture", command=lambda : checkers_utils.choose_white(computer_cam))
white_button.pack()
start_button = ttk.Button(root, text="Start The Game", command=root.destroy)
start_button.grid(row = 2, pady = 20)
root.mainloop()
computer_cam.release()
# GUI end

# starting the game
pygame.init()
window_size = (640, 640)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Checkers")

checkers = Checkers(screen)
checkers.main(window_size[0], window_size[1])






