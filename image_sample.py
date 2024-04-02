import cv2
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import board_utils

class Camera_API:
    def __init__(self,video=False):
        current_script_dir = os.path.dirname(os.path.abspath(__file__))

        # Initialize camera
        if not video:
            self.checkers_cam = cv2.VideoCapture(0)
            self.computer_cam = cv2.VideoCapture(1)

        else:
            print("using video")
            self.camara_xy_path = os.path.join(current_script_dir, "video", "vid1xy.mp4")
            #self.camara_yz_path = os.path.join(current_script_dir, "video", "vid1z.mp4")

            #self.camara_xy_path = 1
            #self.camara_yz_path = 2
            self.cam_xy = cv2.VideoCapture(self.camara_xy_path)
            #self.cam_z = cv2.VideoCapture(self.camara_yz_path)

    def stream_video(self, ref_img, camera = "xy", save_frame = "new_pic", verbose = False, record = False):
        """Stream video from the camera"""
        if camera == "xy" and self.checkers_cam.isOpened():
            cap = self.checkers_cam
        elif camera == "z" and self.computer_cam.isOpened():
            cap = self.computer_cam
        res = 0
        reset_flag = 0
        curr_holo_mat = None
        counter = 0

        if record:
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
			int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
 
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(save_frame + "_video" + '.avi', fourcc, 30, size)

        while True:
            ret, frame = cap.read()
            # print(f"reset_flag: {reset_flag}")
            # print(f"current res is: {res}")
            if ret:
                res, aligned_frame, curr_holo_mat,_, _, aligned_img_clean = board_utils.get_locations(frame, ref_img, curr_holo_mat, reset_flag,
                                                                              verbose=verbose)
                if res != 0:
                    cv2.putText(frame, "Couldn't find the board's inner corners", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.imshow('frame', frame)
                    out.write(frame)
                    #write text on the frame of "couldn't align the frame"

                    reset_flag = 1


                else:
                    frame = aligned_frame
                    cv2.imshow('frame', frame)
                    out.write(frame)
                    reset_flag = 0

                    
                if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to exit
                    break
                elif cv2.waitKey(1) & 0xFF == ord('s'): #press 's' to save frame
                    self.save_frame(aligned_img_clean, save_frame +  "_" + str(counter))
                    print(f"Frame saved as {save_frame + '_' + str(counter)}.jpg")
                    counter += 1
                elif cv2.waitKey(1) & 0xFF == ord('r'):
                    reset_flag = 1
                    print("A reset request has been made")
                



            else:
                print(f"Error: Failed to capture frame from camera {camera}.")
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def record_video(self, video_name, camera = "xy"):
        """Record video from the camera"""
        if camera == "xy" and self.checkers_cam.isOpened():
            cap = self.checkers_cam
        elif camera == "z" and self.computer_cam.isOpened():
            cap = self.computer_cam
        else:
            print("Error: Camera is not opened.")
            return


        # Define the codec and create VideoWriter object
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
			int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
 
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(video_name + '.mp4', fourcc, 30, size)

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                out.write(frame)
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                    break

        cap.release()
        out.release()
        cv2.destroyAllWindows()



    def save_frame(self, frame, save_name='frame'):
        """saves a frame to a file in the images_taken directory"""
        Path.mkdir(Path.cwd() / "images_taken" , parents=True, exist_ok=True)
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite( "images_taken/" + save_name + ".jpg", frame)


    def open_cameras(self, camera_number_xy=0, camera_number_z=1):
        if not self.cam_xy.isOpened():
            self.cam_xy.open(camera_number_xy)
        else:
            print("Error: Camera XY is already opened.")

        if not self.cam_z.isOpened():
            self.cam_z.open(camera_number_z)
        else:
            print("Error: Camera Z is already opened.")

    def open_camera_checkers_cam(self, cam_num=0):

        """Open connections to the specified cameras"""
        if not self.checkers_cam.isOpened():
            self.checkers_cam.open(cam_num)
        else:
            print("Error: Checkers camera is already opened.")


    def open_camera_computer_cam(self, cam_num=1):
        """Open connections to the specified cameras"""
        if not self.computer_cam.isOpened():
            self.computer_cam.open(cam_num)
        else:
            print("Error: Computer camera is already opened.")



######################################### Ref utils #########################################

    def get_frames_with_skip(self, skip_count):
        """
        Reads frames from the video capture, skipping a specified number of frames.

        :param cap: The video capture object.
        :param skip_count: Number of frames to skip.
        :return: The next frame after skipping, or None if there is no frame.
        """
        for _ in range(skip_count + 1):  # Skip 'skip_count' frames
            frame1 = self.read_frame('xy')
            frame2 = self.read_frame('z')
        return (frame1,frame2)  # Return the next frame after skipping
    def close_cameras(self):
        """Release both cameras when finished"""
        if self.cam_xy.isOpened():
            self.cam_xy.release()

        if self.cam_z.isOpened():
            self.cam_z.release()

    def close_computer_cam(self):
        if self.computer_cam.isOpened():
            self.computer_cam.release()

    def close_checkers_cam(self):
        if self.checkers_cam.isOpened():
            self.checkers_cam.release()

    def read_frame(self, cam="checkers"):
        """Read and return the current frame from the specified camera"""
        if cam == "checkers":
            cap = self.checkers_cam
        else:
            cap = self.computer_cam
        ret, frame = cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            print(f"Error: Failed to capture frame from camera.")
            return None

    def read_frames(self, camera='xy', num_frames=1):
        """Read and return multiple frames from the specified camera"""
        frames = []
        for _ in range(num_frames):
            frame = self.read_frame(camera=camera)
            if frame is not None:
                frames.append(frame)
        return frames

    def read_both_frames(self):
        frame_xy = self.read_frame(camera='xy')
        frame_z = self.read_frame(camera='z')
        return frame_xy,frame_z
    def display_frame(self, frame, window_name='Camera Feed'):
        """Display a frame in a window"""
        cv2.imshow(window_name, frame)
        cv2.waitKey(1300)
    


    def close_display_window(self, window_name='Camera Feed'):
        """Close the display window"""
        cv2.destroyWindow(window_name)

    def select_rectangle(self,matrix):
        # Create a window and display the matrix as an image
        cv2.namedWindow('Select Rectangle', cv2.WINDOW_NORMAL)
        cv2.imshow('Select Rectangle', matrix)

        # Variables to store rectangle coordinates
        rectangle_points = []

        # Mouse callback function to capture mouse events
        def mouse_callback(event, x, y, flags, param):
            nonlocal rectangle_points

            if event == cv2.EVENT_LBUTTONDOWN:
                rectangle_points.append((x, y))
                if len(rectangle_points) == 4:
                    cv2.destroyWindow('Select Rectangle')  # Close the window after 4 points are selected

        # Set the mouse callback function
        cv2.setMouseCallback('Select Rectangle', mouse_callback)

        # Wait for the user to select four points
        while len(rectangle_points) < 4:
            cv2.waitKey(10)
        cv2.destroyWindow('Select Rectangle')
        return rectangle_points