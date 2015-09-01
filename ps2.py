"""Problem Set 2: Edges and Lines."""

import numpy as np
import cv2

import os
from math import pi
from math import atan2

input_dir = "input"  # read images from os.path.join(input_dir, <filename>)
output_dir = "output"  # write images to os.path.join(output_dir, <filename>)

def show_image(image):
    #Helper function to show images
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def hough_lines_acc(img_edges, rho_res=1, theta_res=pi/90):
    """Compute Hough Transform for lines on edge image.

    Parameters
    ----------
        img_edges: binary edge image
        rho_res: rho resolution (in pixels)
        theta_res: theta resolution (in radians)

    Returns
    -------
        H: Hough accumulator array
        rho: vector of rho values, one for each row of H
        theta: vector of theta values, one for each column of H
    """
    # TODO: Your code here
    height, width = img_edges.shape[:2]
    theta = np.arange(-90*(pi/180),90*(pi/180),theta_res)
    diagonal = np.sqrt(np.square(float(height))+np.square(float(width)))
    rho = np.arange(0.0, diagonal, rho_res, dtype=np.float)
    H = np.zeros((rho.size, theta.size), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            for index,_theta in enumerate(theta):
                _rho = x*np.cos(_theta) + y*np.sin(_theta)
                rho_index = np.where(rho == int(_rho))
                if rho_index[0]:
                    H[rho_index[0][0],index] += 1


    return H, rho, theta


def hough_peaks(H, Q):
    """Find peaks (local maxima) in accumulator array.

    Parameters
    ----------
        H: Hough accumulator array
        Q: number of peaks to find (max)

    Returns
    -------
        peaks: Px2 matrix (P <= Q) where each row is a (rho_idx, theta_idx) pair
    """
    # TODO: Your code here
    return peaks


def hough_lines_draw(img_out, peaks, rho, theta):
    """Draw lines on an image corresponding to accumulator peaks.

    Parameters
    ----------
        img_out: 3-channel (color) image
        peaks: Px2 matrix where each row is a (rho_idx, theta_idx) index pair
        rho: vector of rho values, such that rho[rho_idx] is a valid rho value
        theta: vector of theta values, such that theta[theta_idx] is a valid theta value
    """
    pass  # TODO: Your code here (nothing to return, just draw on img_out directly)


def main():
    """Run code/call functions to solve problems."""

    # 1-a
    # Load the input grayscale image
    img = cv2.imread(os.path.join(input_dir, 'ps2-input0.png'), 0)  # flags=0 ensures grayscale

    # TODO: Compute edge image (img_edges)
    img_edges = cv2.Canny(img,100,200)
    cv2.imwrite(os.path.join(output_dir, 'ps2-1-a-1.png'), img_edges)  # save as ps2-1-a-1.png
    show_image(img_edges)

    # 2-a
    # Compute Hough Transform for lines on edge image
    H, rho, theta = hough_lines_acc(img_edges)  # TODO: implement this, try calling with different parameters

    # TODO: Store accumulator array (H) as ps2-2-a-1.png
    # Note: Write a normalized uint8 version, mapping min value to 0 and max to 255
    show_image(H)

    # 2-b
    # Find peaks (local maxima) in accumulator array
    peaks = hough_peaks(H, 10)  # TODO: implement this, try different parameters

    # TODO: Store a copy of accumulator array image (from 2-a), with peaks highlighted, as ps2-2-b-1.png

    # 2-c
    # Draw lines corresponding to accumulator peaks
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # copy & convert to color image
    hough_lines_draw(img_out, peaks, rho, theta)  # TODO: implement this
    cv2.imwrite(os.path.join(output_dir, 'ps2-2-c-1.png'), img_out)  # save as ps2-2-c-1.png

    # 3-a
    # TODO: Read ps2-input0-noise.png, compute smoothed image using a Gaussian filter

    # 3-b
    # TODO: Compute binary edge images for both original image and smoothed version

    # 3-c
    # TODO: Apply Hough methods to smoothed image, tweak parameters to find best lines

    # 4
    # TODO: Like problem 3 above, but using ps2-input1.png

    # 5
    # TODO: Implement Hough Transform for circles

    # 6
    # TODO: Find lines a more realtistic image, ps2-input2.png

    # 7
    # TODO: Find circles in the same realtistic image, ps2-input2.png

    # 8
    # TODO: Find lines and circles in distorted image, ps2-input3.png


if __name__ == "__main__":
    main()
