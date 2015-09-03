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

def find_my_bin(array, value):
    """A helper function to find the closest bin the value would fit in

    Parameters
    ----------
        array: the array of bins
        value: the value to fit

    Returns
    -------
        indx: the index of the bin
    """
    value = abs(value)
    idx = (np.abs(array-value)).argmin()

    return idx

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
    # get the edge image height and width
    height, width = img_edges.shape[:2]
    # create the array of thetas. The index into this array will corespond to
    # that in the accumulator
    theta = np.arange(-pi/2.0,pi/2.0+theta_res, theta_res)
    # get the maximum r of the system
    diagonal = int(np.sqrt(np.square(float(height))+np.square(float(width))))
    # set up array of rhos, step size is equal to 1/rho_res
    rho = np.arange(0.0, diagonal, rho_res,dtype=np.float)
    # initialize the accumulator to zero, with dimensions  =
    # rho.size x theta.size
    H = np.zeros((rho.size, theta.size), dtype=np.uint8)
    # build the accumulator
    for y in range(height):
        for x in range(width):
            # only look at pixels that are not black
            if img_edges[y,x] > 0:
                if x+1 < width and y+1 < height:
                    d_x = abs(np.int32(img_edges[y,x+1]) - np.int32(img_edges[y,x]))
                    d_y = abs(np.int32(img_edges[y+1,x]) - np.int32(img_edges[y,x]))

                _theta = atan2(d_y, d_x)
                _rho = x*np.cos(_theta) + y*np.sin(_theta)
                rho_idx = find_my_bin(rho,_rho)
                theta_idx = find_my_bin(theta, _theta)
                H[rho_idx, theta_idx] += 1
    H /= H.max()/255.0
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
    peaks = []  # list of peaks
    maxes = []  # list of maxes
    # index of peak and maxes is one to one
    height, width = H.shape[:2] # get the height and width of the accumulator
    max_value = H.max()
    for rho in range(height):
        for theta in range(width):
            h_val = H[rho,theta]
            if h_val >= 100:
                # check if we have reached the max number of peaks
                maxes_len = len(maxes)
                if maxes_len >= Q:
                    # check to see if the val we are looking at is bigger than
                    # any we already have
                    smallest_idx = maxes.index(min(maxes))
                    if maxes[smallest_idx] < h_val:
                        # get rid of the smallest value
                        maxes.pop(smallest_idx)
                        peaks.pop(smallest_idx)
                        # add the new value
                        maxes.append(h_val)
                        peaks.append([rho,theta])

                else:
                    # add rho and theta to the peaks list
                    maxes.append(h_val)
                    peaks.append([rho,theta])

    return peaks

def draw_box_on_peaks(H, peaks):
    for rho_idx, theta_idx in peaks:
        cv2.rectangle(H, (theta_idx-2, rho_idx-2),(theta_idx+2, rho_idx+2),(0,255,0),1)

def hough_lines_draw(img_out, peaks, rho, theta):
    """Draw lines on an image corresponding to accumulator peaks.

    Parameters
    ----------
        img_out: 3-channel (color) image
        peaks: Px2 matrix where each row is a (rho_idx, theta_idx) index pair
        rho: vector of rho values, such that rho[rho_idx] is a valid rho value
        theta: vector of theta values, such that theta[theta_idx] is a valid theta value
    """
    img_size = img_out.size
    for rho_idx,theta_idx in peaks:
        _rho = rho[rho_idx]
        _theta = theta[theta_idx]
        a = np.cos(_theta)
        b = np.sin(_theta)
        x0 = a*_rho
        y0 = b*_rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img_out, (x1,y1), (x2,y2), (0,255,0),1)


def hough_circles_acc(img_edges, radi_range):
    """Compute Hough Transform for circles

    Parameters
    ----------
        img_edges: binary edge image
        dp: inverse ratio of accumulator to image
        minDist: the minimum distance between two circles
        radi_range: the target radius to search for.

    Returns
    -------
        H: Hough Circle accumulator array
    """
    height, width = img_edges.shape[:2]
    if type(radi_range) is tuple:
        minRadius = min(radi_range)
        maxRadius = max(radi_range)
        possible_radii = range(minRadius, maxRadius)
    else:
        possible_radii = [radi_range]
    gradients = np.arange(-pi/2, pi/2, pi/180)
    H = np.zeros((height, width, len(possible_radii)), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            if img_edges[y,x] > 0:
                for r_index,radius in enumerate(possible_radii):
                    for theta in gradients:
                        a = x-radius*np.cos(theta)
                        b = y-radius*np.sin(theta)
                        if a < height and b < width:
                            H[a,b,r_index] += 1

    return H
def main():
    """Run code/call functions to solve problems."""

    # 1-a
    # Load the input grayscale image
    img = cv2.imread(os.path.join(input_dir, 'ps2-input0.png'), 0)  # flags=0 ensures grayscale

    # TODO: Compute edge image (img_edges)
    img_edges = cv2.Canny(img,0,255)
    cv2.imwrite(os.path.join(output_dir, 'ps2-1-a-1.png'), img_edges)  # save as ps2-1-a-1.png

    # 2-a
    # Compute Hough Transform for lines on edge image
    H, rho, theta = hough_lines_acc(img_edges, 1, pi/180)  # TODO: implement this, try calling with different parameters

    # TODO: Store accumulator array (H) as ps2-2-a-1.png
    # Note: Write a normalized uint8 version, mapping min value to 0 and max to 255
    cv2.imwrite(os.path.join(output_dir, 'ps2-2-a-1.png'), H)

    # 2-b
    # Find peaks (local maxima) in accumulator array
    peaks = hough_peaks(H, 10)  # TODO: implement this, try different parameters

    # TODO: Store a copy of accumulator array image (from 2-a), with peaks
    # highlighted, as ps2-2-b-1.pngH
    H_out = cv2.cvtColor(H, cv2.COLOR_GRAY2BGR)
    draw_box_on_peaks(H_out, peaks)
    cv2.imwrite(os.path.join(output_dir, 'ps2-2-b-1.png'), H_out)

    # 2-c
    # Draw lines corresponding to accumulator peaks
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # copy & convert to color image
    hough_lines_draw(img_out, peaks, rho, theta)  # TODO: implement this
    cv2.imwrite(os.path.join(output_dir, 'ps2-2-c-1.png'), img_out)  # save as ps2-2-c-1.png

    # 3-a
    # TODO: Read ps2-input0-noise.png, compute smoothed image using a Gaussian filter
    noisy_img = cv2.imread(os.path.join(input_dir, 'ps2-input0-noise.png'), 0)   # flags=0 ensures grayscale
    gaussian_blurred_img = cv2.GaussianBlur(noisy_img,(5,5),3)
    cv2.imwrite(os.path.join(output_dir, 'ps2-3-a-1.png'), gaussian_blurred_img)

    # 3-b
    # TODO: Compute binary edge images for both original image and smoothed version
    noisy_img_edges = cv2.Canny(noisy_img,0,255)
    gb_img_edges = cv2.Canny(gaussian_blurred_img,150,255)
    cv2.imwrite(os.path.join(output_dir, 'ps2-3-b-1.png'), noisy_img_edges)
    cv2.imwrite(os.path.join(output_dir, 'ps2-3-b-2.png'), gb_img_edges)

    # 3-c
    # TODO: Apply Hough methods to smoothed image, tweak parameters to find best lines
    H, rho, theta = hough_lines_acc(gb_img_edges, 3, pi/270)
    peaks = hough_peaks(H, 50)
    H_out = cv2.cvtColor(H, cv2.COLOR_GRAY2BGR) # copy & convert to color
    noisy_img_out = cv2.cvtColor(noisy_img, cv2.COLOR_GRAY2BGR) # copy & convert to color
    draw_box_on_peaks(H_out, peaks)
    hough_lines_draw(noisy_img_out, peaks, rho, theta)
    cv2.imwrite(os.path.join(output_dir, 'ps2-3-c-1.png'), H_out)
    cv2.imwrite(os.path.join(output_dir, 'ps2-3-c-2.png'), noisy_img_out)

    # 4
    # TODO: Like problem 3 above, but using ps2-input1.png
    img = cv2.imread(os.path.join(input_dir, 'ps2-input1.png'), 0)   # flags=0 ensures grayscale
    gb_img = cv2.GaussianBlur(img, (5,5),2)
    cv2.imwrite(os.path.join(output_dir, 'ps2-4-a-1.png'), gb_img)
    img_edges = cv2.Canny(gb_img, 0, 255)
    cv2.imwrite(os.path.join(output_dir, 'ps2-4-b-1.png'), img_edges)
    H, rho, theta = hough_lines_acc(img_edges, 0.25, pi/1080)
    peaks = hough_peaks(H, 5)
    H_out = cv2.cvtColor(H, cv2.COLOR_GRAY2BGR)
    draw_box_on_peaks(H_out, peaks)
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    hough_lines_draw(img_out, peaks, rho, theta)
    cv2.imwrite(os.path.join(output_dir, 'ps2-4-c-1.png'), H_out)
    cv2.imwrite(os.path.join(output_dir, 'ps2-4-c-2.png'), img_out)
    #show_image(img_out)

    # 5
    # TODO: Implement Hough Transform for circles
    H = hough_circles_acc(img_edges, 20)
    print H
    H_out = cv2.cvtColor(H, cv2.COLOR_GRAY2BGR)
    show_image(H_out)
    

    # 6
    # TODO: Find lines a more realtistic image, ps2-input2.png

    # 7
    # TODO: Find circles in the same realtistic image, ps2-input2.png

    # 8
    # TODO: Find lines and circles in distorted image, ps2-input3.png


if __name__ == "__main__":
    main()
