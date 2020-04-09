import argparse
import os
import os.path as osp
import pickle as pkl
import random
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from darknet import Darknet
from util import *

# Creating commands lind arguments


def arg_parse():
    """Parse arguments to detect module

    """
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    # To specify the input images or directory of images
    parser.add_argument("--images", dest='images', help="Image / Directory containing images to perform detection upon",
                        default="imgs", type=str)
    # Directory to save detections to
    parser.add_argument("--det", dest='det', help="Image / Directory to store detections to",
                        default="det", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence",
                        help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh",
                        help="NMS Threshhold", default=0.4)
    # Alternative configuration file
    parser.add_argument("--cfg", dest='cfgfile', help="Config file",
                        default="cfg/yolov3.cfg", type=str)
    # Alternative weight file
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile",
                        default="weights/yolov3.weights", type=str)
    # Input image's resolution, can be used for speed-accuracy tradoff
    parser.add_argument("--reso", dest='reso', help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)

    return parser.parse_args()


def write_bbox(x, results, classes, colors):
    """Draw a rectangle with a colour. Also creates a filled rectangle on the top left corner with the class of the object detected in the bounding box.

    Args:
        x (tensor): prediction table with resized coordinates.
        results (list): list of image paths.
        classes (list): list of class names
        colors: just colors...

    Returns:
        img (numpy array): image written with bboxes.
    """
    corner1 = tuple(x[1:3].int())
    corner2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{}".format(classes[cls])
    cv2.rectangle(img, corner1, corner2, color, thickness=tl)
    tf = max(tl - 1, 1)  # font thickness
    t_size = cv2.getTextSize(label, 0, fontScale=tl/3, thickness=tf)[0]
    corner2 = corner1[0] + t_size[0]+3, corner1[1]-t_size[1]-3
    cv2.rectangle(img, corner1, corner2, color, -1)
    cv2.putText(img, label, (corner1[0], corner1[1]-2),
                0, tl/3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


if __name__ == '__main__':
    args = arg_parse()
    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thresh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()

    # Load the classes
    num_classes = 80    # COCO
    classes = load_classes('data/coco.names')

    # Set up the neural network
    print("Loading network...")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded!")

    model.net_info['height'] = args.reso
    input_dim = int(model.net_info['height'])
    assert input_dim % 32 == 0
    assert input_dim > 32

    if CUDA:
        model.cuda()

    # Set the model in evaluation mode
    model.eval()

    read_dir = time.time()  # Checkpoint
    # Detection phase
    try:
        # imglist stores the image/images paths
        imglist = [osp.join(osp.realpath('.'), images, img)
                   for img in os.listdir(images)]
    except NotADirectoryError:
        imglist = []
        imglist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print("No file or directory with the name {}".format(images))
        exit()

    if not os.path.exists(args.det):
        os.makedirs(args.det)

    # Load images
    load_batch = time.time()    # Checkpoint
    loaded_imgs = [cv2.imread(x) for x in imglist]   # numpy array, BGR

    # PyTorch variables for images
    img_batches = list(map(prep_image, loaded_imgs, [
                       input_dim for x in range(len(imglist))]))

    # List containing dimensions of original images
    img_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_imgs]
    img_dim_list = torch.FloatTensor(img_dim_list).repeat(1, 2)

    # Create the batches
    # Every batch will be a concatenation of <=batch_size images
    # The batches are stored in a list.
    leftover = 0
    if(len(img_dim_list) % batch_size):
        leftover = 1
    if batch_size != 1:
        num_batches = len(imglist) // batch_size + leftover
        img_batches = [torch.cat(
            (img_batches[i*batch_size:min((i+1)*batch_size,
                                          len(img_batches))]), dim=0) for i in range(num_batches)]

    if CUDA:
        img_dim_list = img_dim_list.cuda()

    # Detection loop
    write = 0
    start_det_loop = time.time()    # Checkpoint
    for i, batch in enumerate(img_batches):
        # Load the image
        start = time.time()  # Checkpoint
        if CUDA:
            batch = batch.cuda()
        with torch.no_grad():   # This is not training
            prediction = model(batch, CUDA)

        prediction = write_results(
            prediction=prediction,
            confidence=confidence,
            num_classes=num_classes,
            nms_conf=nms_thresh
        )
        end = time.time()   # Checkpoint

        if type(prediction) == int:   # No detection for this batch
            for img_index, image in enumerate(
                    imglist[i*batch_size:min((i+1)*batch_size, len(imglist))]):
                # Denote image path and processing time
                print("{0:20s} predicted in {1:6.3f} seconds".format(
                    image.split("/")[-1], (end - start)/batch_size))
                # No object detected
                print("{0:20s} {1:s}".format("Objects Detected:", ""))
                print("----------------------------------------------------------")
            continue    # No detection for this batch, so move to next batch directly

        # Bbox indices as global image indices
        prediction[:, 0] += i*batch_size

        if not write:
            output = prediction
            write = 1
        else:
            output = torch.cat((output, prediction))    # Default dim = 0

        for img_index, image in enumerate(
                imglist[i*batch_size:min((i+1)*batch_size, len(imglist))]):
            global_img_index = i*batch_size + img_index  # Image index in all images
            objs = [classes[int(x[-1])]  # Class names of this image in a list
                    for x in output if int(x[0]) == global_img_index]
            print("{0:20s} predicted in {1:6.3f} seconds".format(
                image.split("/")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
            print("----------------------------------------------------------")

        # Make sure that CUDA kernel is synchronized with the CPU
        if CUDA:
            torch.cuda.synchronize()

    # Draw bounding boxes on images
    try:
        output
    except NameError:
        print("No detections were made")
        exit()

    # Fit the corner bbox coordinates to insert image field
    # Only choose those images with detections
    img_dim_list = torch.index_select(img_dim_list, 0, output[:, 0].long())
    scaling_factors = torch.min(input_dim/img_dim_list, 1)[0].view(-1, 1)

    # x of corners
    output[:, [1, 3]] -= (input_dim-scaling_factors *
                          img_dim_list[:, 0].view(-1, 1))/2
    # y of corners
    output[:, [2, 4]] -= (input_dim-scaling_factors *
                          img_dim_list[:, 1].view(-1, 1))/2

    # Resizing to original sizes
    output[:, 1:5] /= scaling_factors

    # Clip any bounding boxes that may have boundaries outside the image to the edges of our image
    for i in range(output.shape[0]):
        output[i, [1, 3]] = torch.clamp(
            output[i, [1, 3]], 0.0, img_dim_list[i, 0])
        output[i, [2, 4]] = torch.clamp(
            output[i, [2, 4]], 0.0, img_dim_list[i, 1])

    # Many colors from pickled file
    output_recast = time.time()  # Checkpoint
    class_load = time.time()    # Checkpoint
    colors = pkl.load(open('pallete', 'rb'))

    draw = time.time()
    # Iteration
    list(map(lambda x: write_bbox(x, loaded_imgs, classes, colors), output))

    # Detection images paths
    det_names = pd.Series(imglist).apply(
        lambda x: "{}/det_{}".format(args.det, x.split('/')[-1]))

    # Write the images with detections to the addresses in det_names
    list(map(cv2.imwrite, det_names, loaded_imgs))
    end = time.time()   # Checkpoint

    print("SUMMARY")
    print("----------------------------------------------------------")
    print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
    print()
    print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
    print("{:25s}: {:2.3f}".format(
        "Loading batch", start_det_loop - load_batch))
    print("{:25s}: {:2.3f}".format(
        "Detection (" + str(len(imglist)) + " images)", output_recast - start_det_loop))
    print("{:25s}: {:2.3f}".format(
        "Output Processing", class_load - output_recast))
    print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
    print("{:25s}: {:2.3f}".format(
        "Average time_per_img", (end - load_batch)/len(imglist)))
    print("----------------------------------------------------------")
    torch.cuda.empty_cache()
