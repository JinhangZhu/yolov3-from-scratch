"""
Various helper functions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


def unique(indices):
    """Get classes present in any given image, since there can be multiple tru detections of the same class.

    Parameter:
        indices (tensor): the indices of max values of different bboxes.

    Returns:
        indices_res (tensor): the indices of the unique max values of different bboxes.

    Example:
        indices = tensor([0, 0, 1])
        indices_res = tensor([0, 1])
    """
    indices_np = indices.cpu().numpy()
    unique_np = np.unique(indices_np)
    unique_indices = torch.from_numpy(unique_np)

    indices_res = unique_indices.detach().clone()
    return indices_res


def bbox_iou(box1, box2):
    """Calculates the IoU of two bounding boxes.

    Args:
        box1 (2D tensor): bounding box 1
        box2 (2D tensor): bounding box 2

    Returns:
        iou (2D tensor): IoU of the boxes
    # """
    inter_max_xy = torch.min(box1[:, 2:4], box2[:, 2:4])
    inter_min_xy = torch.max(box1[:, 0:2], box2[:, 0:2])

    inter_size = torch.clamp((inter_max_xy-inter_min_xy), min=0)
    inter_area = inter_size[:, 0]*inter_size[:, 1]  # 1 by num_boxes

    # # Original codes
    # # Get the coordinates of bounding boxes
    # b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    # b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # # Get the coordinates of the intersection rectangle
    # # Can take multi-row tensors
    # inter_rect_x1 = torch.max(b1_x1, b2_x1)
    # inter_rect_y1 = torch.max(b1_y1, b2_y1)
    # inter_rect_x2 = torch.min(b1_x2, b2_x2)
    # inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # # Intersection area
    # inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
    #     torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    # # Union Area
    # b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    # b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)

    b1_area = (box1[:, 2]-box1[:, 0])*(box1[:, 3]-box1[:, 1])
    b2_area = (box2[:, 2]-box2[:, 0])*(box2[:, 3]-box2[:, 1])

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


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


    Args:
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
    print("Predictions: ", prediction)

    return prediction


def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    """To obtain the true detections via subjecting the output to objectness score thresholding and Non-Maximum Suppression.

    Args:
        prediction (tensor):    Prediction table of bboxes
        confidence ():          Obejctness score threshold
        num_classes (int):      Would be 80 in COCO case
        nms_conf (float):       The NMS IoU threshold

    Returns:
        output (tensor):        Shape (D * 8), where D is the number of true detections in all of images, each presented by a row. Each detection has 8 attributes, anemly, index of the image in the batch to which the detection belongs to, 4 corner coordinates, objectness score, the score of the class with maximum confidence, and the index of the class.
    """
    # 1. Objectness confidence thresholding
    # < threshold   ->  all zeros in this bbox
    conf_mask = (prediction[:, :, 4] > confidence).float(
    ).unsqueeze(2)  # Same number of dims as prediction
    prediction *= conf_mask

    # 2. Localise the left-top and right-bottom corners
    box_corner = prediction.detach().clone()
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2]/2)
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3]/2)
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2]/2)
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3]/2)
    prediction[:, :, :4] = box_corner[:, :, :4]

    batch_size = prediction.size(0)
    write = False   # output not initialised

    # 3. Loop over images of the batch
    for ib in range(batch_size):
        image_prediction = prediction[ib]    # Only 2D tensor now

        # 3.1. Confidence thresholding
        # Only concerned with the class score having the maximum value
        # Remove the 80 class scored from each row, and instead add the
        # index of the class having the maximum values, as well as the score.

        max_conf, max_conf_indices = torch.max(
            image_prediction[:, 5:5+num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)    # .float() might be removed
        max_conf_indices = max_conf_indices.float().unsqueeze(1)
        image_prediction = torch.cat(
            (image_prediction[:, :5], max_conf, max_conf_indices), 1)

        # Get rid of rows with zero objectness
        non_zero_indices = torch.nonzero(image_prediction[:, 4]).squeeze()
        image_prediction_ = image_prediction[non_zero_indices, :].view(-1, 7)

        if image_prediction_.shape[0] == 0:
            continue    # End current iteration, move to the nest image

        # # Original codes--------------
        # non_zero_ind = (torch.nonzero(image_pred[:, 4]))
        # try:
        #     image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
        # except:
        #     continue

        # # For PyTorch 0.4 compatibility
        # # Since the above code with not raise exception for no detection
        # # as scalars are supported in PyTorch 0.4
        # if image_pred_.shape[0] == 0:
        #     continue
        # # ------------

        # Get the various classes detected in the image
        img_classes = unique(image_prediction_[:, -1])

        # 3.2. Classwise NMS
        for cls in img_classes:
            # 3.2.1 Get detections assigned to the current class
            cls_mask = image_prediction_ * \
                (image_prediction_[:, -1] == cls).float().unsqueeze(1)
            cls_mask_indices = torch.nonzero(cls_mask[:, -2]).squeeze()
            img_pred_classes = image_prediction_[
                cls_mask_indices].view(-1, 7)  # The bboxes with the same class

            # 3.2.2 Sort the detections in the sequence of objectness score from highest on the top to the lowest on the bottom
            obj_conf_desc_indices = torch.sort(
                img_pred_classes[:, 4], descending=True)[1]
            img_pred_classes = img_pred_classes[obj_conf_desc_indices]
            num_detections = img_pred_classes.size(0)

            # 3.2.3 NMS
            for i in range(num_detections):
                # Get the IoUs of all boxes below the one that we are looing at
                try:
                    ious = bbox_iou(img_pred_classes[i].unsqueeze(
                        0), img_pred_classes[i+1:])
                except ValueError:
                    break   # second input param -> empty tensor
                except IndexError:
                    break   # index out of bound

                # Zero out all the detections that have IoU > threshold, i.e. similar to the top box
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                img_pred_classes[i+1:] *= iou_mask

                # Keep the non-zero rows, including bboxes that are very distinct from the top one
                non_zero_indices = torch.nonzero(
                    img_pred_classes[:, 4]).squeeze()
                img_pred_classes = img_pred_classes[non_zero_indices].view(
                    -1, 7)

            # 3.2.4 Writing the predictions
            # e.g. for batch with index: i, having k detections, the batch_indices will be a k-by-1 tensor filled with i.
            batch_indices = img_pred_classes.new_full(
                (img_pred_classes.size(0), 1), ib)
            # Original:
            # batch_indices = img_pred_classes.new(img_pred_classes.size(0), 1).fill_(ib)
            seq = batch_indices, img_pred_classes   # tuple

            # Detections concatenation
            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                new_out = torch.cat(seq, 1)
                output = torch.cat((output, new_out))

    try:
        return output
    except:
        return 0    # Not a single detection in the batch


def load_classes(namesfile):
    """Load the classes.name file.

    Args:
        namefile (string): namesfile path.

    Returns:
        names (list): A list of class names.
    """
    fp = open(namesfile, 'r')
    names = fp.read().split('\n')[:-1]
    return names


def letterbox_image(source, input_dim):
    """Resize image with unchanged aspect ratio using padding.

    Args:
        source (numpy array): The source image.
        input_dim (tuple): (height, width) of the input image of the model.

    Returns:
        canvas (numpy array): The resized image with unchanged ratio.
    """
    src_height, src_width = source.shape[0], source.shape[1]
    input_height, input_width = input_dim
    multiple = min(input_height/src_height, input_width/src_width)
    dst_height = int(src_height * multiple)
    dst_width = int(src_width*multiple)
    resized_image = cv2.resize(
        source, (dst_width, dst_height), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((input_height, input_width, 3), 128)
    canvas[(input_height-dst_height)//2:(input_height-dst_height)//2 + dst_height,
           (input_width-dst_width)//2: (input_width-dst_width)//2 + dst_width, :] = resized_image

    return canvas


def prep_image(img, input_dim):
    """Prepare image for inputting to the neural network. Transforming from numpy to tensor.

    Args:
        img (numpy array): The numpy array image.
        input_dim (int): (height, width) of the input image of the model.

    Returns:
        img (tensor): Image in tensor type.
    """
    img = (letterbox_image(img, (input_dim, input_dim)))
    # BGR -> RGB, HWC -> CHW
    img = img[:, :, :: -1].transpose((2, 0, 1)).copy()
    # numpy -> tensor, normalise, CHW -> BCHW
    img = torch.from_numpy(img).float().div(255).unsqueeze(0)
    return img
