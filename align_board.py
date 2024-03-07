import image_sample
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils

def align_board(img_to_al, ref_img, max_features = 300, keep_percent = 1, verbose = False, crop_width = 70, crop_height = 140):
    """
    Aligns the board in the image to the reference image.
    :param img_to_al: The image to align.
    :param ref_img: The reference image.
    :param max_features: The maximum number of features to detect in the images.
    :param keep_percent: The percentage of features to keep when aligning the images.
    :param verbose: Whether to display the images and print the results.
    :return: The aligned image.
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
    aligned_img = aligned_img[crop_height:height-crop_height, crop_width:width-crop_width] # Crop the image
    # Display the images and print the results
    if verbose:
        print(f"Number of matches: {len(matches)}")

        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].imshow(img_matches)
        ax[0].set_title('Matches')
        ax[1].imshow(aligned_img)
        ax[1].set_title('Aligned Image')
        plt.show()

    return aligned_img


if __name__ == "__main__":
    img_to_al = plt.imread("images_taken/img_6.jpg")
    ref_img = plt.imread("images_taken/ref_img.jpg")

    verbose = True
    max_lines = 20

    aligned_img = align_board(img_to_al, ref_img, verbose=True)
    aligned_img_gray = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
    #apply canny edge detection
    edges = cv2.Canny(aligned_img_gray, 50, 400)
    #compute the hough lines
    lines = cv2.HoughLines(edges, 2, 3*np.pi / 180, 200)
    
    # #find the intersections of the lines[:max_lines]
    # intersections = []
    # for i in range(len(lines[:max_lines])):
    #     for j in range(i + 1, len(lines[:max_lines])):
    #         rho1, theta1 = lines[i][0]
    #         rho2, theta2 = lines[j][0]
    #         A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
    #         b = np.array([[rho1], [rho2]])
    #         x0, y0 = np.linalg.solve(A, b)
    #         x0, y0 = int(np.round(x0)), int(np.round(y0))
    #         intersections.append((x0, y0))
    # #draw the intersections on the image
    # for point in intersections:
    #     cv2.circle(aligned_img, point, 10, (0, 0, 255), 3)
    
    
    # draw the lines on the image
    for line in lines[:max_lines]:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 2000 * (-b))
        y1 = int(y0 + 2000 * (a))
        x2 = int(x0 - 2000 * (-b))
        y2 = int(y0 - 2000 * (a))
        cv2.line(aligned_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    #compute the intesections of the lines
    
    if verbose:
        #display the image
        plt.imshow(aligned_img, cmap='gray')
        plt.show()