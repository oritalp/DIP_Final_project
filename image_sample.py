import cv2
import os
import numpy as np
class Camera_API:
    def _init_(self,video=False):
        current_script_dir = os.path.dirname(os.path.abspath(_file_))

        # Initialize camera
        if not video:
            self.cam_xy = cv2.VideoCapture(1)
            self.cam_z = cv2.VideoCapture(2)

        else:
            print("using video")
            self.camara_xy_path = os.path.join(current_script_dir, "video_light", "vid1xy.mp4")
            self.camara_yz_path = os.path.join(current_script_dir, "video_light", "vid1z.mp4")

            #self.camara_xy_path = 1
            #self.camara_yz_path = 2
            self.cam_xy = cv2.VideoCapture(self.camara_xy_path)
            self.cam_z = cv2.VideoCapture(self.camara_yz_path)

    def open_camera(self, camera_number_xy=0, camera_number_z=1):
        """Open connections to the specified cameras"""
        if not self.cam_xy.isOpened():
            self.cam_xy.open(camera_number_xy)
        else:
            print("Error: Camera XY is already opened.")

        if not self.cam_z.isOpened():
            self.cam_z.open(camera_number_z)
        else:
            print("Error: Camera Z is already opened.")

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

    def read_frame(self, camera='xy'):
        """Read and return the current frame from the specified camera"""
        if camera == 'xy':
            cap = self.cam_xy
        elif camera == 'z':
            cap = self.cam_z
        else:
            print("Error: Invalid camera selection.")
            return None

        ret, frame = cap.read()
        if ret:
            return np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            print(f"Error: Failed to capture frame from camera {camera}.")
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
        cv2.waitKey(1)

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