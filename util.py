"""
Various helper functions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


def predict_transform(prediction, input_dim, anchors, num_classes, CUDA=True):
    """Takes an detection feature map and turns it into a 2D tensor, where each row of the tensor corresponds to attributes of a bounding box. The 2D tensor is like:

    BBox per cell   ->  index in (batch, number of cells * number of anchors per cell, number of attributes)
    1st BBox        ->  (0, 000, 5+num_classes)
    2nd BBox        ->  (0, 001, 5+num_classes)
    3rd BBox        ->  (0, 002, 5+num_classes)
    1st BBox        ->  (0, 010, 5+num_classes)
    2nd BBox        ->  (0, 011, 5+num_classes)
    3rd BBox        ->  (0, 012, 5+num_classes)
    ...
    1st BBox        ->  (bs-1, [gs-1][gs-1]0, 5+num_classes)
    2nd BBox        ->  (bs-1, [gs-1][gs-1]1, 5+num_classes)
    3rd BBox        ->  (bs-1, [gs-1][gs-1]2, 5+num_classes)


    Parameters:
        prediction (tensor): previous output
        input_dim (int): input image dimension
        anchors (list(tuple)): used anchors in this yolo layer
        num_classes (int): total number of classes of the dataset
        CUDA (bool): optional CUDA flag, default: True

    Returns:
        prediction (tensor): Resized (3D tensor) prediction output of this yolo layer. The tensor is in three dimensions:  [batch size, number of bounding boxes, attributes of the bounding box]

        Batch size: batch_size
        Number of bounding boxes: number of cells * number of anchors per cell
        Number of attributes: 5 + number of classes, attributes including centre_X, centre_Y, bbox_height, bbox_width, objectness score, class 0 score, class 1 score, ...
    """
    batch_size = prediction.size(0)
    stride = input_dim // prediction.size(2)
    grid_size = input_dim // stride
    bbox_attributes = 5+num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(
        batch_size, bbox_attributes*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(
        batch_size, grid_size*grid_size*num_anchors, bbox_attributes)

    # Proportionally resize the anchors by the value of stride
    # [(,),(,),(,)] ->  tensor([[,],[,],[,]])  size([3,2])
    anchors = [(anchor[0]/stride, anchor[1]/stride) for anchor in anchors]

    # Sigmoid transform: centre_X, centre_Y, objectness score
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # Add the grid offsets to the centre coordinates
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)  # Same row numer as prediction
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
        prediction = prediction.cuda()  # Reminded by zhihu comment

    # IMPORTANT
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(  # Rescale by num_anchors
        1, num_anchors).view(-1, 2).unsqueeze(0)  # unqueeze not necessary
    prediction[:, :, :2] += x_y_offset

    # Apply the anchors to the dimensions of the bounding boxes
    # log space transform on height and width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)

    # IMPORTANT
    prediction[:, :, 2:4] = torch.exp(
        prediction[:, :, 2:4])*anchors    # Elementwise mulplication

    # Sigmoid activation of class scores
    prediction[:, :, 5:5 +
               num_classes] = torch.sigmoid(prediction[:, :, 5:5+num_classes])

    # Resize the prediction map to the size of the imput image
    prediction[:, :, :4] *= stride

    return prediction
