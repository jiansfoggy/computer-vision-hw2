#!/usr/bin/env python
'''
A module for applying a steerable filter on an image.

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
from scipy.misc import imsave
import numpy as np

from eta.core.config import Config
import eta.core.image as etai
import eta.core.module as etam


class SteerableFilterConfig(etam.BaseModuleConfig):
    '''Steerable filter configuration settings.

    Attributes:
        data (DataConfig)
    '''

    def __init__(self, d):
        super(SteerableFilterConfig, self).__init__(d)
        self.data = self.parse_object_array(d, "data", DataConfig)


class DataConfig(Config):
    '''Data configuration settings.

    Inputs:
        sobel_horizontal_result (eta.core.types.NpzFile): The result of
            convolving the original image with the "sobel_horizontal" kernel.
            This will give the value of Gx (the gradient in the x
            direction).
        sobel_vertical_result (eta.core.types.NpzFile): The result of
            convolving the original image with the "sobel_vertical" kernel.
            This will give the value of Gy (the gradient in the y
            direction).
    Outputs:
        filtered_image (eta.core.types.ImageFile): The output image after
            applying the steerable filter.
    '''

    def __init__(self, d):
        self.sobel_horizontal_result = self.parse_string(
            d, "sobel_horizontal_result")
        self.sobel_vertical_result = self.parse_string(
            d, "sobel_vertical_result")
        self.filtered_image = self.parse_string(
            d, "filtered_image")


def _apply_steerable_filter(G_x, G_y):
    '''Applies the steerable filter on the given image, using the results
    from sobel kernel convolution.

    Args:
        G_x: the x derivative of the image, given as the result of convolving
            the input image with the horizontal sobel kernel
        G_y: the y derivative of the image, given as the result of convolving
            the input image with the vertical sobel kernel

    Returns:
        g_intensity: the intensity of the input image, defined as
            sqrt(Gx^2 + Gy^2), at every line that does not lie on the
            x-axis or y-axis
    '''
    g_intensity = np.zeros(G_x.shape)

    # Take arctan, conver to degrees, round to integer, take absolute value, store in orientation matrix
    orientation = np.abs(np.rint(np.arctan2(G_y, G_x) * (180 / np.pi)))

    for i in range(G_x.shape[0]):
        for j in range(G_x.shape[1]):
            if np.mod(orientation[i,j], 90) > 5:	# Removes orientations very close to vert/horiz
                g_intensity[i,j] = np.sqrt((G_x[i,j]**2) + (G_y[i,j]**2))

    # imsave('gradient.jpg', g_intensity)
    # imsave('Gx.jpg', G_x)
    # imsave('Gy.jpg', G_y)
    return g_intensity


def _filter_image(steerable_filter_config):
    for data in steerable_filter_config.data:
        sobel_horiz = np.load(data.sobel_horizontal_result)["filtered_matrix"]
        sobel_vert = np.load(data.sobel_vertical_result)["filtered_matrix"]
        filtered_image = _apply_steerable_filter(sobel_horiz, sobel_vert)
        etai.write(filtered_image, data.filtered_image)


def run(config_path, pipeline_config_path=None):
    '''Run the Steerable Filter module.

    Args:
        config_path: path to a SteerableFilterConfig file
        pipeline_config_path: optional path to a PipelineConfig file
    '''
    steerable_filter_config = SteerableFilterConfig.from_json(config_path)
    etam.setup(steerable_filter_config,
               pipeline_config_path=pipeline_config_path)
    _filter_image(steerable_filter_config)


if __name__ == "__main__":
    run(*sys.argv[1:])
