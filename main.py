import cv2
import os
import numpy as np
from image_sample import Camera_API


#camera_api = Camera_API()

#camera_api.open_camera()
#frame = camera_api.read_frame()
#camera_api.display_frame(frame)
#camera_api.close_display_window()
#camera_api.close_cameras()

path = os.getcwd()
print(os.path.isfile(path+"/bord.png"))
image = cv2.imread(path+"/bord.png")
cv2.imshow(image)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
 
# Use canny edge detection
edges = cv2.Canny(gray,50,150,apertureSize=3)
 
# Apply HoughLinesP method to 
# to directly obtain line end points
lines_list =[]
lines = cv2.HoughLinesP(
            edges, # Input edge image
            1, # Distance resolution in pixels
            np.pi/180, # Angle resolution in radians
            threshold=100, # Min number of votes for valid line
            minLineLength=5, # Min allowed length of line
            maxLineGap=10 # Max allowed gap between line for joining them
            )
 
# Iterate over points
for points in lines:
      # Extracted points nested in the list
    x1,y1,x2,y2=points[0]
    # Draw the lines joing the points
    # On the original image
    cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)
    # Maintain a simples lookup list for points
    lines_list.append([(x1,y1),(x2,y2)])
     
# Save the result image
cv2.imwrite('detectedLines.png',image)

print("done")

