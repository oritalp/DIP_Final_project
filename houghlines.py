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
image = cv2.imread(path+"/bord.jpg")

# making a binary pictur
lwr = np.array([0, 0, 0])
upr = np.array([150, 150, 150]) #TODO: need to find optimal valus
msk = cv2.inRange(image, lwr, upr)
cv2.imshow("msk", msk)
krn = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 30))
dlt = cv2.dilate(msk, krn, iterations=5)
res = 255 - cv2.bitwise_and(dlt, msk)

# Use canny edge detection
edges = cv2.Canny(res, 40, 150, None, 3)
edges = cv2.dilate(edges, None, iterations=3)

# Apply HoughLinesP method to 
# to directly obtain line end points
lines_list =[]
lines = cv2.HoughLinesP(
            edges, # Input edge image
            1, # Distance resolution in pixels
            np.pi/180, # Angle resolution in radians
            threshold=800, # Min number of votes for valid line
            minLineLength=800, # Min allowed length of line
            maxLineGap=30 # Max allowed gap between line for joining them
            )
 
# Iterate over points and lines to the image
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
