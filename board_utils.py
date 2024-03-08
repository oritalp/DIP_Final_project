import image_sample
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils

def align_board(img_to_al, ref_img, max_features=300, keep_percent=1,
                 verbose=False, crop_width=70, crop_height=140):
    """
    Aligns the input image to a reference image using feature matching and homography.

    Parameters:
    - img_to_al: The image to be aligned (numpy array).
    - ref_img: The reference image (numpy array).
    - max_features: The maximum number of features to detect (default: 300).
    - keep_percent: The percentage of best matches to keep (default: 1).
    - verbose: Whether to display images for debug purposes(default: False).
    - crop_width: The width to crop from the aligned image (default: 70).
    - crop_height: The height to crop from the aligned image (default: 140).

    Returns:
    a tuple containing:
    - aligned_img: The aligned image (numpy array).
    - h: The homography matrix (numpy array).
    """

    # Convert the images to grayscale
    img_to_al_gray = cv2.cvtColor(img_to_al, cv2.COLOR_BGR2GRAY)
    ref_img_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute the descriptors
    orb = cv2.ORB_create(max_features)
    kp1, des1 = orb.detectAndCompute(img_to_al_gray, None)
    kp2, des2 = orb.detectAndCompute(ref_img_gray, None)

    # Match the features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort the matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Keep only the best matches
    num_good_matches = int(len(matches) * keep_percent)
    matches = matches[:num_good_matches]

    # Draw the matches
    img_matches = cv2.drawMatches(img_to_al, kp1, ref_img, kp2, matches, None)
    img_matches = imutils.resize(img_matches, width=2000)

    # Extract the matched keypoints
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    # Find the homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use the homography to align the images
    height, width, _ = ref_img.shape
    aligned_img = cv2.warpPerspective(img_to_al, h, (width, height))
    aligned_img = aligned_img[crop_height:height-crop_height, crop_width:width-crop_width]  # Crop the image

    # Display the images and print the results
    if verbose:
        print(f"Number of matches: {len(matches)}")
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].imshow(img_matches)
        ax[0].set_title('Matches')
        ax[1].imshow(aligned_img)
        ax[1].set_title('Aligned Image')
        plt.show()

    return aligned_img,h

def get_intersections(img, max_lines=14, crop_width=70, crop_height=70,
                      rho_reso=3, theta_reso=3*np.pi/180, verbose=False):
    """
    Detects and returns the intersections of lines in an image.

    Parameters:
    - img: The input image.
    - max_lines: The maximum number of lines to consider for intersection detection.
    - crop_width: The width of the image to be cropped before processing.
    - crop_height: The height of the image to be cropped before processing.
    - rho_reso: The resolution parameter for the HoughLines function.
    - theta_reso: The resolution parameter for the HoughLines function.
    - verbose: If True, displays the image with detected intersections and prints the results.

    Returns:
    - intersections: A list of (x, y) coordinates representing the detected intersections.

    Raises:
    - ValueError: If the number of detected intersections is not equal to 49.
    """

    img_crp = img[crop_height:img.shape[0]-crop_height, crop_width:img.shape[1]-crop_width] # Crop the image
    # Convert the image to grayscale
    img_crp_gray = cv2.cvtColor(img_crp, cv2.COLOR_BGR2GRAY)
    # Apply Canny edge detection
    edges = cv2.Canny(img_crp_gray, 50, 400)
    # Compute the Hough lines
    lines = cv2.HoughLines(edges, rho_reso, theta_reso, 200)
    lines = lines[:max_lines]
    # Find the intersections of the lines[:max_lines]
    intersections = []
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
            intersections.append((x0+crop_width, y0+crop_height))
    # Draw the intersections on the image
    for point in intersections:
        cv2.circle(img, point, 10, (0, 0, 255), 3)
    # Display the image and print the results
    if verbose:
        intersections = sorted(intersections, key=lambda x: x[0])
        intersections = sorted(intersections, key=lambda x: x[1])
        print(intersections)
        print(f"Number of points: {len(intersections)}")
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(img)
        ax.set_title('Intersections')
        plt.show()

    if len(intersections) != 49:
        return ValueError("Only {len(intersections)} intersections found. Expected 49.")
    
    return intersections

#TODO: think how to cluster close points in intersections

if __name__ == "__main__":
    img_to_al = plt.imread("images_taken/img_6.jpg")
    ref_img = plt.imread("images_taken/ref_img.jpg")

    verbose = True
    max_lines = 14
    crop_width = 70
    crop_height = 70

    aligned_img,_ = align_board(img_to_al, ref_img, verbose=False)
    intersect = get_intersections(aligned_img, verbose=True)
