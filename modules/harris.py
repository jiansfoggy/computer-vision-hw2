#!/usr/bin/env python
'''
A module for determining the locations of corners in an image using
the Harris Corner Detection algorithm.

Info:
    type: eta.core.types.Module
    version: 0.1.0
'''
# pragma pylint: disable=redefined-builtin
# pragma pylint: disable=unused-wildcard-import
# pragma pylint: disable=wildcard-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *
from collections import defaultdict
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import sys

import numpy as np
import cv2
from scipy import signal

from eta.core.config import Config, ConfigError
import eta.core.image as etai
import eta.core.module as etam
import eta.core.utils as etau


class HarrisCornerConfig(etam.BaseModuleConfig):
    '''Harris corner configuration settings.

    Attributes:
        data (DataConfig)
        parameters (ParametersConfig)
    '''

    def __init__(self, d):
        super(HarrisCornerConfig, self).__init__(d)
        self.data = self.parse_object_array(d, "data", DataConfig)
        self.parameters = self.parse_object(d, "parameters", ParametersConfig)
        self._validate()

    def _validate(self):
        '''Ensures the right inputs are given to the module before
        actual computation begins.
        '''
        for data in self.data:
            if ((data.corners_img_after_sup or data.corners_img_before_sup)
                    and not data.input_image):
                raise ConfigError(
                    "ERROR! Original image required for visualization.")


class DataConfig(Config):
    '''Data configuration settings.

    Inputs:
        input_image (eta.core.types.Image): [None] The input image
        sobel_horizontal_result (eta.core.types.NpzFile): The result of
            convolving the original image with the "sobel_horizontal" kernel.
            This will give the value of Gx (the gradient in the x
            direction).
        sobel_vertical_result (eta.core.types.NpzFile): The result of
            convolving the original image with the "sobel_vertical" kernel.
            This will give the value of Gy (the gradient in the y
            direction).
    Outputs:
        corner_locations (eta.core.types.NpzFile): [None] The location of every
            corner detected by the Harris Corner algorithm, sorted by
            confidence
        corners_img_before_sup (eta.core.types.ImageFile): [None] The corner
            response image before suppression
        corners_img_after_sup (eta.core.types.ImageFile): [None] The corner
            response image after suppression
    '''

    def __init__(self, d):
        self.input_image = self.parse_string(
            d, "input_image", default=None)
        self.sobel_horizontal_result = self.parse_string(
            d, "sobel_horizontal_result")
        self.sobel_vertical_result = self.parse_string(
            d, "sobel_vertical_result")
        self.corner_locations = self.parse_string(
            d, "corner_locations", default=None)
        self.corners_img_before_sup = self.parse_string(
            d, "corners_img_before_sup", default=None)
        self.corners_img_after_sup = self.parse_string(
            d, "corners_img_after_sup", default=None)


class ParametersConfig(Config):
    '''Parameter configuration settings.

    Parameters:
        window_half_size (eta.core.types.Number): [7] The half-size of the window
        threshold (eta.core.types.Number): [0.001] The lower-bound threshold for
            corner "confidence".
        non_max_suppression_radius (eta.core.types.Number): [2] The
            circular radius used when applying non-maximum suppression to the
            corner response.
    '''

    def __init__(self, d):
        self.window_half_size = self.parse_number(
            d, "window_half_size", default=7)
        self.threshold = self.parse_number(
            d, "threshold", default=0.001)
        self.non_max_radius = self.parse_number(
            d, "non_max_suppression_radius", default=2)


def _get_harris_corner(Gx, Gy, window_half_size, threshold):
    '''Performs Harris corner detection. The output will be a matrix, the
    same size as Gx and Gy, where every element (pixel) contains the
    corner strength at that location. The corner strength is defined to
    be the minimum eigenvalue of the structure tensor at that pixel
    location.

    Args:
        Gx: The x-derivative of the image, which is the output of convolving
            with the horizontal sobel kernel
        Gy: the y-derivative of the image, which is the output of convolving
            with the vertical sobel kernel
        window_half_size: the half-size of the window to consider when computing
            the structure tensor
        threshold: the lower-bound threshold to use when choosing corners

    Returns:
        corner_response_matrix: a matrix, the same size as Gx and Gy, where
            every element (pixel) contains the corner strength at that
            location.
        corner_location_matrix: an Nx2 matrix, where every row is the location
            of a corner whose response is greater than the threshold. The matrix
            should also be sorted by corner strength, such that the corner
            with the highest corner strength is put in row 0 and the corner
            with the lowest corner strength is put in row N-1.
    '''
    h11 = 0
    h22 = 0
    h_cross = 0
    corner_response_matrix = np.zeros(Gx.shape)

    col_zeros = np.zeros((Gx.shape[0], window_half_size))		# column of zeros
    row_zeros = np.zeros((window_half_size, Gx.shape[1] + (window_half_size * 2)))	# row of zeros
    padded_matrix = np.hstack((col_zeros, corner_response_matrix, col_zeros))
    padded_matrix = np.vstack((row_zeros, padded_matrix, row_zeros))
    # print("col_zeros = ", col_zeros.shape)
    # print("row_zeros = ", row_zeros.shape)
    # print("padded matrix size =", padded_matrix.shape)

    Gx_2 = np.square(Gx)		# square Gx and Gy via
    Gy_2 = np.square(Gy)		# element-wise squaring
    
    for i in range(0, corner_response_matrix.shape[0]):
        for j in range(0, corner_response_matrix.shape[1]):
            temp_X = Gx_2[i-window_half_size:i+window_half_size+1, j-window_half_size:j+window_half_size+1]
            temp_Y = Gy_2[i-window_half_size:i+window_half_size+1, j-window_half_size:j+window_half_size+1]
            h11 = np.sum(temp_X)		# sum of squared elements in window of Gx
            h22 = np.sum(temp_Y)		# sum of squared elements in window of Gy
            h_cross = np.sum(np.multiply(temp_X, temp_Y))		# element-wise multiplication

            H = np.array([[h11, h_cross], [h_cross, h22]])	# 'H' is structure tensor
            vals = np.linalg.eigvals(H)

            if min(vals) > threshold:            
                corner_response_matrix[i,j] = min(vals)		# store eigenval if greater than threshold
            else:
                corner_response_matrix[i,j] = 0

    c = np.where(corner_response_matrix > 0)	# returns indices at located corners
    eigens = corner_response_matrix[c]		# (2xN) matrix - eigenvalues at each index
    eigens = np.reshape(eigens, (1, np.shape(eigens)[0]))	# (1xN) matrix
    stack_it = np.vstack((eigens, c))		# (3xN) matrix
    stack_it = np.transpose(stack_it)		# (Nx3) matrix
    sorted_indices = np.sort(stack_it, axis=0)[::-1]	# sorts by eigenvalue - [::-1] makes it descending order
    corner_location_matrix = sorted_indices[:,1:3]	# location matrix is just (x,y) coords.

    # print("Size of Gx = ",Gx.shape)
    # print("Size of Gy = ",Gy.shape)
    # print("Size of corner response = ",corner_response_matrix.shape)
    # print("Size of location response = ",corner_location_matrix.shape)
    etai.write(corner_response_matrix*255, "Response.png") 
    return corner_response_matrix, corner_location_matrix


def non_max_suppression(corner_response, radius):
    """Finds corners in a given response map.

    This method uses a circular region to define the non-maxima suppression
    area. For example, let c1 be a corner representing a peak in the Harris
    response map, any corners in the area determined by the circle of radius
    'radius' centered in c1 should not be returned in the peaks array.
    Make sure you account for duplicate and overlapping points.

    Args:
        corner_response: floating-point response map,
            e.g. output from the Harris detector.
        radius: radius of circular region for non-maximal suppression.

    Returns:
        corners_sup: peaks found in response map R, each row must be defined
            as [x, y]. Array size must be N x 2, where N are the number of
            points found.
    """
    r_map1 = np.copy(corner_response)
    data_max = np.zeros(r_map1.shape)
    ind = np.nonzero(r_map1)

    for n in range(len(ind[0])):
        i = ind[0][n]
        j = ind[1][n]
        frame = r_map1[max((i-radius), 0):min((i+radius),r_map1.shape[0]),
                       max((j-radius),0):min((j+radius),r_map1.shape[1])]
        if r_map1 [i,j]< np.max(frame): # not local max
            data_max [i,j]  = 0
        elif np.max(data_max[
                max((i-radius), 0):min((i+radius),data_max.shape[0]),
                max((j-radius),0):min((j+radius),data_max.shape[1])])>0:
            # Tie, and already as a max t
            data_max [i,j]  = 0
        else:
            data_max [i,j] = r_map1[i, j]

    col_ind, row_ind = np.nonzero(data_max)
    # You can use the distance as a conditional measure to merge the points.
    # Average of the x and y coordinates of the close points to merge.
    corners = []
    for i in range(len(row_ind)):
        corners.append((row_ind[i], col_ind[i]))
    corners = tuple(corners)

    return np.array(corners)


def _visualize_corners(in_img, corner_locations):
    '''Creates an image that shows the detected corners.

    Args:
        in_img: the original image
        corner_locations: the locations of each detected corner in the
            image, stored in an Nx2 matrix

    Returns:
        out_img: an image with the detected corners colored in red

    '''
    if etai.is_gray(in_img):
        out_img = etai.gray_to_rgb(in_img)
    else:
        out_img = in_img.copy()
    for i in range(corner_locations.shape[0]):
        cv2.circle(out_img,
                   (int(corner_locations[i][0]), int(corner_locations[i][1])),
                   2,
                   (255,0,0),
                   -1)
    return out_img


def _find_corners(harris_corner_config):
    for data in harris_corner_config.data:
        sobel_horiz = np.load(data.sobel_horizontal_result)["filtered_matrix"]
        sobel_vert = np.load(data.sobel_vertical_result)["filtered_matrix"]
        corner_response, corner_locations = _get_harris_corner(
                                sobel_horiz, sobel_vert,
                                harris_corner_config.parameters.window_half_size,
                                harris_corner_config.parameters.threshold)
        corner_locs_after_sup = non_max_suppression(
            corner_response, harris_corner_config.parameters.non_max_radius)
        if data.corner_locations:
            etau.ensure_basedir(data.corner_locations)
            np.savez(data.corner_locations,
                     corner_locations=corner_locs_after_sup)
        if data.corners_img_before_sup or data.corners_img_after_sup:
            in_img = etai.read(data.input_image)
            if data.corners_img_before_sup:
                corners_viz_before_sup = _visualize_corners(in_img,
                    corner_locations)
                etai.write(corners_viz_before_sup, data.corners_img_before_sup)
            if data.corners_img_after_sup:
                corners_viz_after_sup = _visualize_corners(in_img,
                    corner_locs_after_sup)
                etai.write(corners_viz_after_sup, data.corners_img_after_sup)


def run(config_path, pipeline_config_path=None):
    '''Run the Harris Corner module.

    Args:
        config_path: path to a HarrisCornerConfig file
        pipeline_config_path: optional path to a PipelineConfig file
    '''
    harris_corner_config = HarrisCornerConfig.from_json(config_path)
    etam.setup(harris_corner_config, pipeline_config_path=pipeline_config_path)
    _find_corners(harris_corner_config)


if __name__ == "__main__":
    run(*sys.argv[1:])
