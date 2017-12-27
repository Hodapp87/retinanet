"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import keras.backend
from .dynamic import meshgrid

import numpy as np


def bbox_transform_inv(boxes, deltas, mean=None, std=None):
    """Computes object box predictions from the bounding-box regression
    values.  Returns a NumPy array the same same as 'boxes' or
    'deltas', but with each row giving [x0,y0,x1,y1] of the adjusted
    bounding box (adjusted by the respective row in 'deltas').

    This performs the inverse of bbox_transform. It parametrizes
    bounding boxes according to appendix C of the Fast R-CNN paper
    (arXiv 1311.2524v5), specifically, equations 1-4.

    Parameters:
    boxes -- Array giving anchor coordinates as rows of [x0,y0,x1,y1]
    deltas -- Array giving box regression values as [tx, ty, tw, th].
              Should be same shape as 'boxes'.
    mean -- Optional 4-element array with respective means for [tx, ty, tw, th]
    std -- Optional 4-element array with standard deviations (same format)

    """
    if mean is None:
        mean = [0, 0, 0, 0]
    if std is None:
        std = [0.1, 0.1, 0.2, 0.2]

    widths  = boxes[:, :, 2] - boxes[:, :, 0]
    heights = boxes[:, :, 3] - boxes[:, :, 1]
    ctr_x   = boxes[:, :, 0] + 0.5 * widths
    ctr_y   = boxes[:, :, 1] + 0.5 * heights

    dx = deltas[:, :, 0] * std[0] + mean[0]
    dy = deltas[:, :, 1] * std[1] + mean[1]
    dw = deltas[:, :, 2] * std[2] + mean[2]
    dh = deltas[:, :, 3] * std[3] + mean[3]

    pred_ctr_x = ctr_x + dx * widths
    pred_ctr_y = ctr_y + dy * heights
    pred_w     = keras.backend.exp(dw) * widths
    pred_h     = keras.backend.exp(dh) * heights

    pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
    pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
    pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
    pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

    pred_boxes = keras.backend.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], axis=2)

    return pred_boxes


def shift(shape, stride, anchors):
    """Shifts base 'relative' anchors by the appropriate amount for a
    feature map's stride size and shape.  This produces a Keras
    variable instance containing the coordinates (in the input image's
    coordinate space) for each anchor corresponding to each sliding
    window location, with shape of [shape[0]*shape[1]*num_anchors, 4].

    Parameters:
    shape -- Tuple/list containing (y,x) dimensions of the feature map.
    stride -- Distance in the input image for each step in the feature map
    anchors -- Keras variable giving anchors relative to reference window
               (e.g. 9 total for 3 scales and 3 aspect ratios). Shape should
               be (num_anchors, 4), with each row being (x0,y0,x1,y1) relative
               coordinates.
    """
    # ([0,1,2,3..., X-1] + [0.5,0.5,0.5...])*stride =
    # [0.5, 1.5, 2.5, ... (X-0.5)]*stride
    shift_x = (keras.backend.arange(0, shape[1], dtype=keras.backend.floatx()) +
               keras.backend.constant(0.5, dtype=keras.backend.floatx())) * stride
    # Likewise for Y:
    shift_y = (keras.backend.arange(0, shape[0], dtype=keras.backend.floatx()) +
               keras.backend.constant(0.5, dtype=keras.backend.floatx())) * stride

    shift_x, shift_y = meshgrid(shift_x, shift_y)
    shift_x = keras.backend.reshape(shift_x, [-1])
    shift_y = keras.backend.reshape(shift_y, [-1])

    # Below is columns of (x,y,x,y), where (x,y) is the center point, in
    # image coordinates, of an anchor.  This is then given for every
    # point in the feature map - thus shape[0]*shape[1] columns.
    shifts = keras.backend.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)

    # Switch to shape[0]*shape[1] rows of (x,y,x,y):
    shifts            = keras.backend.transpose(shifts)
    number_of_anchors = keras.backend.shape(anchors)[0]

    # number of base points = shape[0] * shape[1]:
    k = keras.backend.shape(shifts)[0]

    # (1) Add an initial dimension to the base anchors.
    # (2) Add dimension to 'shifts' in the middle.

    # (3) Add the two of them to broadcast add, repeating base anchors
    # and shifts to produce shape [k, number_of_anchors, 4] - thus,
    # shifted anchors for each anchor at each feature map location.
    shifted_anchors = keras.backend.reshape(anchors, [1, number_of_anchors, 4]) + keras.backend.cast(keras.backend.reshape(shifts, [k, 1, 4]), keras.backend.floatx())
    # (why is this cast to floatx when it should be that already?)

    # Merge those first two dimensions.  Thus, rows [0, 1, ...,
    # number_of_anchors-1] are coordinates for the first feature map
    # location, [number_of_anchors, ..., (2*number_of_anchors-1)] are
    # the second, and so on.
    shifted_anchors = keras.backend.reshape(shifted_anchors, [k * number_of_anchors, 4])

    return shifted_anchors
