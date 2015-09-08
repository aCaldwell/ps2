"""Problem Set 2: Edges and Lines."""

import numpy as np
import cv2

import os
from math import pi
from math import atan2
from math import atan

input_dir = "input"  # read images from os.path.join(input_dir, <filename>)
output_dir = "output"  # write images to os.path.join(output_dir, <filename>)

def write_out_image(img, img_name):
    """ Write out the image with the proper name

    Parameters
    ----------
        img: the image to be written out
        img_name: the name of the image

    """
    cv2.imwrite(os.path.join(output_dir, img_name), img)

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
    # get the height and width of the image
    height, width = img_edges.shape[:2]
    # calculate the diagonal
    diagonal = int(np.sqrt(np.square(height)+np.square(width)))
    # build the array of rhos
    rho = np.arange(0.0, diagonal, rho_res)
    # build the array of thetas
    theta = np.arange(-pi/2.0, pi/2.0+theta_res, theta_res)
    # initialize H to zero
    H = np.zeros((rho.size, theta.size), dtype=np.uint8)
    # find the indices of all the edge pixels
    indices = np.where(img_edges == 255)
    indices = zip(indices[0], indices[1])
    # loop through the indices to build H
    for y_loc, x_loc in indices:
        # delta y and delta x
        d_y = d_x = 0
        if x_loc+1 < width and y_loc+1 < height:
            d_x= abs(np.int32(img_edges[y_loc, x_loc+1]) - np.int32(img_edges[y_loc, x_loc]))
            d_y= abs(np.int32(img_edges[y_loc+1, x_loc]) - np.int32(img_edges[y_loc, x_loc]))
        _theta = atan2(d_y, d_x)
        _rho = int(x_loc*np.cos(_theta) + y_loc*np.sin(_theta))
        theta_idx = find_my_bin(theta, _theta)
        rho_idx = find_my_bin(rho, _rho)
        H[rho_idx, theta_idx] += 1

    # normalize H
    H /= H.max()/255.0

    return H, rho, theta


def hough_peaks(H, Q, threshold=127):
    """Find peaks (local maxima) in accumulator array.

    Parameters
    ----------
                H: Hough accumulator array
                Q: number of peaks to find (max)
        threshold: the lowests acceptable number of votes

    Returns
    -------
        peaks: Px2 matrix (P <= Q) where each row is a (rho_idx, theta_idx) pair
    """
    peaks = []  # list of peaks
    # get the top Q values
    top_q = np.sort(H.flatten())[-Q:][::-1]
    # filter out those that are lower than the threshold
    top_threshold = top_q[top_q >= threshold]
    # build up an indices for the top values
    indices = []
    for value in top_threshold:
        coordinates = zip(*np.where(H == value))
        # skip over it if we already have the coordinate
        if coordinates not in indices:
            indices.append(coordinates)

    # flatten it out to a 1-d list of tuples
    indices = [x for sublist in indices for x in sublist]
    # add them to the peaks
    for rho, theta in indices:
        peaks.append([rho,theta])

    return peaks

def draw_box_on_peaks(H, peaks):
    """Draw boxes around the peaks of the Hough accumulator

    Parameters
    ----------
                H: Hough accumulator array
            peaks: Px2 matrix (P <= Q) where each row is a (rho_idx, theta_idx) pair
    """
    for rho_idx, theta_idx in peaks:
        # Draw a 4x4 box centered around the peak
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


def hough_circles_acc(img_edges, radius, sobel_y=None, sobel_x=None):
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
    H = np.zeros((height, width))
    indices = np.where(img_edges > 0)
    indices = zip(indices[0], indices[1])
    thetas = np.arange(-pi/2-pi/180,pi/2+pi/180,pi/180)
    radius_x_sin_theta = radius*np.sin(thetas)
    radius_x_cos_theta = radius*np.cos(thetas)

    for y, x in indices:
        if sobel_y is None and sobel_x is None:
            for i,a in enumerate(radius_x_sin_theta):
                a = np.round(x+a).astype(np.uint32)
                b = np.round(y+radius_x_cos_theta[i]).astype(np.uint32)
                if a<height and b < width:
                    H[a,b] += 1
        else:
            g_y = sobel_y[y,x]
            g_x = sobel_x[y,x]
            if g_x == 0:
                theta = atan2(g_y,g_x)
            else:
                theta = atan(g_y/g_x)
            a1 = np.round(y-radius*np.sin(theta)).astype(np.uint32)
            b1 = np.round(x-radius*np.cos(theta)).astype(np.uint32)
            a2 = np.round(y+radius*np.sin(theta)).astype(np.uint32)
            b2 = np.round(x+radius*np.cos(theta)).astype(np.uint32)

            if a1 < height and b1 < width:
                H[a1, b1] += 1
            if a2 < height and b2 < width:
                H[a2, b2] += 1

    H /= H.max()/255.0
    return H


def draw_circles(img, circles, radius=20):
    """ Draw circles on the provided image

    Parameters
    ----------
        img: The image that circles will be drawn on
        circles: the circles to draw
        radius: the radius of the circles to be drawn

    """

    for c in circles:
        center = tuple((c[0], c[1]))
        cv2.circle(img, center, 3, (255,0,0), -1, 8, 0)
        cv2.circle(img, center, radius, (0,255,0), 3, 8, 0)


def find_circles(img_edges, radii_range):
    centers = []
    radii = []
    radii_r = np.arange(min(radii_range),max(radii_range))
    for radius in radii_r:
        H = hough_circles_acc(img_edges, radius)
        centers.append(hough_peaks(H, 10, 200))
        radii.append(radius)

    return centers, radii


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
    gb_img_edges = cv2.Canny(gaussian_blurred_img,75,255)
    cv2.imwrite(os.path.join(output_dir, 'ps2-3-b-1.png'), noisy_img_edges)
    cv2.imwrite(os.path.join(output_dir, 'ps2-3-b-2.png'), gb_img_edges)

    # 3-c
    # TODO: Apply Hough methods to smoothed image, tweak parameters to find best lines
    H, rho, theta = hough_lines_acc(gb_img_edges, 3)
    peaks = hough_peaks(H, 6)
    H_out = cv2.cvtColor(H, cv2.COLOR_GRAY2BGR) # copy & convert to color
    noisy_img_out = cv2.cvtColor(noisy_img, cv2.COLOR_GRAY2BGR) # copy & convert to color
    draw_box_on_peaks(H_out, peaks)
    hough_lines_draw(noisy_img_out, peaks, rho, theta)
    cv2.imwrite(os.path.join(output_dir, 'ps2-3-c-1.png'), H_out)
    cv2.imwrite(os.path.join(output_dir, 'ps2-3-c-2.png'), noisy_img_out)

    # 4
    # TODO: Like problem 3 above, but using ps2-input1.png
    img = cv2.imread(os.path.join(input_dir, 'ps2-input1.png'), 0)   # flags=0 ensures grayscale
    gb_img = cv2.GaussianBlur(img, (11,11),4)
    cv2.imwrite(os.path.join(output_dir, 'ps2-4-a-1.png'), gb_img)
    img_edges = cv2.Canny(gb_img, 50, 255)
    cv2.imwrite(os.path.join(output_dir, 'ps2-4-b-1.png'), img_edges)
    H, rho, theta = hough_lines_acc(img_edges, 29, pi/5)
    peaks = hough_peaks(H, 4, 50)
    H_out = cv2.cvtColor(H, cv2.COLOR_GRAY2BGR)
    draw_box_on_peaks(H_out, peaks)
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    hough_lines_draw(img_out, peaks, rho, theta)
    write_out_image(H_out, 'ps2-4-c-1.png')
    write_out_image(img_out, 'ps2-4-c-2.png')

    # 5
    # TODO: Implement Hough Transform for circles
    img_edges = cv2.Canny(gb_img, 0, 90)
    H = hough_circles_acc(img_edges, 20)
    centers = hough_peaks(H, 10, 0)
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    draw_circles(img_out, centers, 20)
    write_out_image(gb_img, 'ps2-5-a-1.png')
    write_out_image(img_edges, 'ps2-5-a-2.png')
    write_out_image(img_out, 'ps2-5-a-3.png')

    centers, radii = find_circles(img_edges, (20,30))
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i, radius in enumerate(radii):
        draw_circles(img_out, centers[i], radius)
    write_out_image(img_out, 'ps2-5-b-1.png')

    # 6
    # TODO: Find lines a more realtistic image, ps2-input2.png
    img = cv2.imread(os.path.join(input_dir, 'ps2-input2.png'), 0)
    gb_img = cv2.GaussianBlur(img, (5,5), 3)
    img_edges = cv2.Canny(gb_img, 50, 110)
    H, rho, theta = hough_lines_acc(img_edges, 30, pi/15)
    peaks = hough_peaks(H, 6, 10)
    H_out = cv2.cvtColor(H, cv2.COLOR_GRAY2BGR)
    draw_box_on_peaks(H_out, peaks)
    img_out = cv2.cvtColor(gb_img, cv2.COLOR_GRAY2BGR)
    hough_lines_draw(img_out, peaks, rho, theta)
    write_out_image(img_out, 'ps2-6-a-1.png')

    # 7
    # TODO: Find circles in the same realtistic image, ps2-input2.png
    gb_img = cv2.GaussianBlur(img, (9,9), 2)
    img_edges = cv2.Canny(gb_img,0,200)
    img_out = cv2.cvtColor(gb_img, cv2.COLOR_GRAY2BGR)
    centers, radii = find_circles(img_edges, (20, 30))
    for i, radius in enumerate(radii):
        draw_circles(img_out, centers[i], radius)

    write_out_image(img_out, 'ps2-7-a-1.png')

    # 8
    # TODO: Find lines and circles in distorted image, ps2-input3.png
    img = cv2.imread(os.path.join(input_dir, 'ps2-input3.png'), 0)
    gb_img = cv2.GaussianBlur(img, (5,5), 2)
    img_edges = cv2.Canny(gb_img, 50, 110)
    H,rho,theta = hough_lines_acc(img_edges, 30, pi/15)
    peaks = hough_peaks(H, 6, 10)
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    hough_lines_draw(img_out, peaks, rho, theta)

    gb_img = cv2.GaussianBlur(img, (9,9), 2)
    img_edges = cv2.Canny(gb_img, 0, 200)
    centers, radii = find_circles(img_edges, (20, 30))
    for i, radius in enumerate(radii):
        draw_circles(img_out, centers[i], radius)

    write_out_image(img_out, 'ps2-8-a-1.png')


if __name__ == "__main__":
    main()
