import image_sample
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
import time
from Path import path


def compute_holo_mat(img_to_al, ref_img, max_features=1000, keep_percent=0.8, num_ignore_first=0,verbose=False):
    """
    Compute the homography matrix for aligning two images using feature matching.

    Args:
        img_to_al (numpy.ndarray): The image to align.
        ref_img (numpy.ndarray): The reference image.
        max_features (int, optional): The maximum number of features to detect. Defaults to 1000.
        keep_percent (float, optional): The percentage of best matches to keep. Defaults to 0.8.
        num_ignore_first (int, optional): The number of initial matches to ignore. Defaults to 0.

    Returns:
        tuple: A tuple containing the result code and the homography matrix.
            - The result code (int) indicates the success or failure of the alignment process.
              A value of 0 indicates success, while a non-zero value indicates failure.
            - The homography matrix (numpy.ndarray) is used to align the images. It is None if the alignment fails.
    """
    res = 0
    # Convert the images to grayscale
    img_to_al_gray = cv2.cvtColor(img_to_al, cv2.COLOR_RGB2GRAY)
    ref_img_gray = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)

    # Detect ORB features and compute the descriptors
    orb = cv2.ORB_create(max_features)
    kp1, des1 = orb.detectAndCompute(img_to_al_gray.astype(np.uint8), None)
    kp2, des2 = orb.detectAndCompute(ref_img_gray.astype(np.uint8), None)

    # Match the features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort the matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Keep only the best matches
    num_good_matches = int(len(matches) * keep_percent)
    matches = matches[num_ignore_first:num_good_matches + num_ignore_first]

    # Draw the matches
    img_matches = cv2.drawMatches(img_to_al, kp1, ref_img, kp2, matches, None)
    img_matches = imutils.resize(img_matches, width=2000)

    # Extract the matched keypoints
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    if verbose:
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.title("ORB Feature Matching")
        plt.show()

    # Find the homography
    try:
        h, _ = cv2.findHomography(points1, points2, cv2.RANSAC)
    except cv2.error:
        res = 1
        return res, None
    else:
        return res, h

def align_board(img_to_al, ref_img, res_holo, h, crop_width=100, crop_height=0):
    """
    Aligns the input image to a reference image using a perspective transformation matrix.

    Args:
        img_to_al (numpy.ndarray): The image to be aligned.
        ref_img (numpy.ndarray): The reference image.
        res_holo (int): The result of the hologram calculation, 0 for success, 1 for failure.
        h (numpy.ndarray): The perspective transformation matrix.
        crop_width (int, optional): The width to crop the aligned image. Defaults to 100.
        crop_height (int, optional): The height to crop the aligned image. Defaults to 0.

    Returns:
        tuple: A tuple containing the result of the alignment (0 for success, 1 for failure) and the aligned image.
    """
    if res_holo != 0:
        return res_holo, img_to_al

    try:
        height, width, _ = ref_img.shape
        aligned_img = cv2.warpPerspective(img_to_al, h, (width, height))
        aligned_img = aligned_img[crop_height:height-crop_height, crop_width:width-crop_width]  # Crop the image

    except cv2.error:
        return 1, img_to_al
    else:
        return 0, aligned_img


def get_intersections(res_align, img, max_lines=14, crop_width_left=40, crop_width_right=60
                      , crop_height_bottom=80, crop_height_top=65,
                      rho_reso=3, theta_reso=5*np.pi/180, verbose=False, hough_threshold=160,
                      canny_low_th = 150, canny_high_th = 400):
    """
    Finds the intersections of lines in an image.

    Parameters:
    - res_align (int): The alignment result. If not equal to 0, the function returns an empty list.
    - img (numpy.ndarray): The input image.
    - max_lines (int): The maximum number of lines to consider.
    - crop_width_left (int): The number of pixels to crop from the left side of the image.
    - crop_width_right (int): The number of pixels to crop from the right side of the image.
    - crop_height_bottom (int): The number of pixels to crop from the bottom of the image.
    - crop_height_top (int): The number of pixels to crop from the top of the image.
    - rho_reso (int): The resolution parameter for the HoughLines function.
    - theta_reso (float): The resolution parameter for the HoughLines function, in radians.
    - verbose (bool): If True, displays the cropped image and the intersections on the original image.

    Returns:
    - intersections (list): A list of tuples representing the x and y coordinates of the intersections,
      if the alignment is successful. if not, returns an empty list.

    """
    intersections = []

    if res_align != 0:
        return intersections
    
    else:

        img_crp = img[crop_height_top:img.shape[0]-crop_height_bottom, crop_width_left:img.shape[1]-crop_width_right] # Crop the image
        # Convert the image to grayscale
        img_crp_gray = cv2.cvtColor(img_crp, cv2.COLOR_BGR2GRAY)
        # Apply Canny edge detection
        edges = cv2.Canny(img_crp_gray, canny_low_th, canny_high_th)
        if verbose:
            plt.imshow(edges, cmap='gray')
            plt.title(f'Canny Edges with low threshold: {canny_low_th} and high threshold: {canny_high_th}')
            plt.show()
        # Compute the Hough lines
        lines = cv2.HoughLines(edges, rho_reso, theta_reso, hough_threshold)
        if lines is None:
            lines = []
        lines = lines[:max_lines]
        if verbose:
            img_crp_copy = cv2.cvtColor(img_crp, cv2.COLOR_BGR2RGB)
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(img_crp_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
            plt.imshow(img_crp_copy)
            plt.title('Detected Lines')
            plt.show()

        # Find the intersections of the lines[:max_lines]
        vertical_lines = [line for line in lines if np.abs(line[0][1]) < 0.1]
        horizontal_lines = [line for line in lines if np.abs(line[0][1] - np.pi/2) < 0.1]
        for horizontal_line in horizontal_lines:
            for vertical_line in vertical_lines:
                rho1, theta1 = horizontal_line[0]
                rho2, theta2 = vertical_line[0]
                A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
                b = np.array([[rho1], [rho2]])
                x0, y0 = np.linalg.solve(A, b)
                x0, y0 = int(np.round(x0)), int(np.round(y0))
                intersections.append((x0+crop_width_left, y0+crop_height_top))
        if verbose:
            #draw the lines on the image
            for line in lines[:max_lines]:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(img_crp, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # Draw the intersections on the image
        if verbose:
            for point in intersections:
                cv2.circle(img, point, 2, (0, 0, 255), 1)
        # Display the image and print the results
        intersections = sorted(intersections, key=lambda x: x[0])
        intersections = sorted(intersections, key=lambda x: x[1])


        if verbose:
            fig, ax = plt.subplots(1,2, figsize=(10, 10))
            ax[0].imshow(cv2.cvtColor(img_crp, cv2.COLOR_BGR2RGB))
            ax[0].set_title('Cropped Image')
            ax[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax[1].set_title('Intersections')
            plt.show()

        # if len(intersections) != 49:
        #     raise ValueError(f"Expected 49 intersections, but found {len(intersections)}.")

        return intersections

def get_circles(img, canny_high_th=60, verbose=False):
    """
    Detects and returns circles in an image.

    Args:
        img (numpy.ndarray): The input image.
        canny_high_th (int, optional): The higher threshold for the Canny edge detector. Defaults to 60.
        verbose (bool, optional): If True, displays the detected circles and Canny edges. Defaults to False.

    Returns:
        list: A list of tuples representing the detected circles. Each tuple contains the x-coordinate, y-coordinate,
              color (0 for blue, 1 for orange), and radius of a circle.

    """
    counter = 0
    output_list = []

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=15,
                               param1=canny_high_th, param2=18, minRadius=10, maxRadius=20)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        circles = circles.squeeze(0)
        for i in circles:
            checked_pixel = img_hsv[i[1] - 1, i[0] - 1, :]
            if (checked_pixel[2] >= 220 and checked_pixel[0] >= 80 and checked_pixel[0] <= 120):
                output_list.append((i[1] - 1, i[0] - 1, 0, i[2]))  # append as white
                counter += 1
            elif (checked_pixel[0] <= 25 or (checked_pixel[0] >= 140 and checked_pixel[0] <= 190)): #the second and third conditions are due to some light chnages that can appear in the image
                output_list.append((i[1] - 1, i[0] - 1, 1, i[2]))  # append as orange
                counter += 1
            else:
                output_list.append((i[1] - 1, i[0] - 1, 2, i[2]))  # append as mistake
            if counter == 24:
                break

    if verbose:
        for i in output_list:
            color = (255, 255, 255) if i[2] == 0 else (0, 0, 0)
            if i[2] == 2:
                color = (0, 255, 0)
            cv2.circle(img, (i[1] - 1, i[0] - 1), i[3], color, 2)
            cv2.circle(img, (i[1] - 1, i[0] - 1), 2, color, 3)
        fig, axes = plt.subplots(1, 2, figsize=(10, 10))
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Circles")
        edges = cv2.Canny(img_gray, canny_high_th // 2, canny_high_th)
        axes[1].imshow(edges, cmap='gray')
        axes[1].set_title("Canny edges")
        plt.show()


    return output_list




def get_locations(img_to_al, ref_img, last_holo_mat, reset_flag=0, min_desc=100, max_desc=1000, keep_percent_min=0.2,
                  keep_percent_max=0.8, canny_high_th=80, verbose=False, verbose_circles=False):
    """
    if reset flag is 0, it uses the last_holo_mat to align the image, if reset flag is 1, it tries to find a new compatible
    holographic matrix using compute_holo_mat function. If it is unable to find such matrix it returns a result code of 1
    (failure). if it finds such a matrix it tries to find circles and intersections. if it meets the criterion it returns
    teh coordinates with positive result, otherwise the result is set to 1.

    Args:
        img_to_al (numpy.ndarray): The image to align.
        ref_img (numpy.ndarray): The reference image.
        last_holo_mat (numpy.ndarray): The last holographic matrix.
        reset_flag (int, optional): if True, tries to find the board from scratch ussing compute_holo_mat with different 
        parameters. deafults to 0.
        min_desc (int, optional): Minimum number of descriptors. Defaults to 100.
        max_desc (int, optional): Maximum number of descriptors. Defaults to 1000.
        keep_percent_min (float, optional): Minimum percentage of keypoints to keep. Defaults to 0.2.
        keep_percent_max (float, optional): Maximum percentage of keypoints to keep. Defaults to 0.8.
        canny_high_th (int, optional): High threshold for Canny edge detection. Defaults to 80.
        verbose (bool, optional): Flag to enable verbose output. Defaults to False.
        verbose_circles (bool, optional): Flag to enable verbose output for circles. Defaults to False.

    Returns:
        tuple: A tuple containing the following elements:
            - result (int): Result code indicating the success or failure of the process.
            - aligned_img (numpy.ndarray): The aligned image.
            - new_holo_mat (numpy.ndarray): The new holographic matrix.
            - intersections (list): List of intersection points.
            - circles (list): List of detected circles.
    """
    start_time = time.time()
    circles = []
    break_flag = False
    result = 0

    if reset_flag or last_holo_mat is None:
        for num_desc in range(min_desc, max_desc, 200):
            for keep_percent in np.arange(keep_percent_min, keep_percent_max, 0.2):
                res_holo, new_holo_mat = compute_holo_mat(img_to_al, ref_img, max_features=num_desc,
                                                          keep_percent=keep_percent)
                res_align, aligned_img = align_board(img_to_al, ref_img, res_holo, new_holo_mat)
                intersections = get_intersections(res_align, aligned_img, verbose=False)
                if check_grid_spacing(intersections):
                    break_flag = True
                    break
            if break_flag:
                break

        else:
            result = 1

    else:
        res, aligned_img = align_board(img_to_al, ref_img, 0, last_holo_mat)
        intersections = get_intersections(res, aligned_img, verbose=False)
        new_holo_mat = last_holo_mat
        if not check_grid_spacing(intersections):
            result = 1

    if break_flag or (reset_flag == 0):
        circles = get_circles(aligned_img, canny_high_th=canny_high_th, verbose=verbose_circles)

    aligned_img_clean = aligned_img.copy()
    
    if verbose:
        for point in intersections:
            cv2.circle(aligned_img, point, 5, (0, 0, 255), 3)
        for i in circles:
            color = (255, 255, 255) if i[2] == 0 else (0, 0, 0)
            if i[2] == 2:
                color = (0, 255, 0)
            # draw the outer circle
            cv2.circle(aligned_img, (i[1], i[0]), i[3], color, 2)
            # draw the center of the circle
            cv2.circle(aligned_img, (i[1], i[0]), 2, color, 3)
            # print(f"Time to process the image: {time.time() - start_time}")


    return result, aligned_img, new_holo_mat, intersections, circles, aligned_img_clean

def reduce_list(lst, tollerance=8):
    """
    Reduces a list of numbers by removing consecutive numbers that are within a given tolerance.
    This is an auxilary function for check_grid_spacing.

    Args:
        lst (list): The input list of numbers.
        tollerance (int, optional): The tolerance value. Defaults to 8.

    Returns:
        list: The reduced list of numbers.
    """

    reduced_list = [lst[0]]
    for i in range(1, len(lst)):
        if lst[i] - reduced_list[-1] > tollerance:
            reduced_list.append(lst[i])
    return reduced_list


def check_grid_spacing(intersections, spacing_tol=10, tol_same_point=8, verbose=False):
    """
    Check if the given intersections form a grid with consistent spacing.
    This is an auxilary function for get_locations to check weather the board is aligned properly.

    Parameters:
    - intersections (list): A list of (x, y) coordinates representing the intersections.
    - spacing_tol (int): Tolerance for the maximum difference in spacing between grid lines.
    - tol_same_point (int): Tolerance for considering two points as the same.
    - verbose (bool): If True, display a scatter plot of the intersections.

    Returns:
    - bool: True if the intersections form a grid with consistent spacing, False otherwise.
    """

    if len(intersections) != 49:
        return False
    
    x_values = [point[0] for point in intersections]
    y_values = [point[1] for point in intersections]

    set_x = sorted(list(set(x_values)))
    set_y = sorted(list(set(y_values)))

    reduced_set_x = reduce_list(set_x, tollerance=tol_same_point)
    reduced_set_y = reduce_list(set_y, tollerance=tol_same_point)

    if len(reduced_set_x) != 7 or len(reduced_set_y) != 7:
        return False

    if verbose:
        plt.scatter(x_values, y_values)
        plt.show()

    distances_x = [reduced_set_x[i] - reduced_set_x[i-1] for i in range(1, len(reduced_set_x))]
    distances_y = [reduced_set_y[i] - reduced_set_y[i-1] for i in range(1, len(reduced_set_y))]

    if max(distances_x) - min(distances_x) > spacing_tol or max(distances_y) - min(distances_y) > spacing_tol:
        return False

    return True

if __name__ == "__main__":
    path = "" # line to Remove in shelli's environment
    img_to_al = cv2.imread(path + "images_taken/clean_pics_0.jpg")
    ref_img = cv2.imread(path + "images_taken/new_alligned.jpg")

    # res, h = compute_holo_mat(img_to_al, ref_img)
    # res_align, aligned_img = align_board(img_to_al, ref_img, res, h)

    # if res_align == 0:
    #     in
    # tersections = get_intersections(res_align, aligned_img,hough_threshold=180, canny_low_th=50, verbose=True,
    #                                        crop_width_left=40, crop_width_right=60, crop_height_bottom=80, crop_height_top=65)
    # else:
    #     print("Error in alignment")

    # camera = image_sample.Camera_API()
    # camera.stream_video(ref_img, camera="z", save_frame="clean_pics", verbose=True)


    min_desc = 100
    max_desc = 1000
    keep_percent_min = 0.2
    keep_percent_max = 0.8
    crop_width_left = 40
    crop_width_right = 60
    crop_height_bottom = 80
    crop_height_top = 65

    # for num_desc in range(min_desc, max_desc, 200):
    #     for keep_percent in np.arange(keep_percent_min, keep_percent_max, 0.2):
    #         res_holo, new_holo_mat = compute_holo_mat(img_to_al, ref_img, max_features=num_desc,
    #                                                     keep_percent=keep_percent)
    #         res_align, aligned_img = align_board(img_to_al, ref_img, res_holo, new_holo_mat)
    #         if res_align ==0:
    #             print(f"num_desc: {num_desc}, keep_percent: {keep_percent}")
    #             cv2.imshow("aligned", aligned_img)
    #             cv2.waitKey(0)

    #700 on 0.8

    res_holo, h = compute_holo_mat(img_to_al, ref_img, max_features=700, keep_percent=0.8)
    res_align, aligned_img = align_board(img_to_al, ref_img, res_holo, h)
    intersections = get_intersections(res_align, aligned_img, crop_width_left=crop_width_left,
                                      crop_width_right=crop_width_right, crop_height_bottom=crop_height_bottom,
                                      crop_height_top=crop_height_top, verbose=True, hough_threshold=100)

    plt.imshow(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB))
    plt.title("Summed up")
    plt.show()



