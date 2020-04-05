"""
Creates the YOLO network.
"""
from util import *
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import Variable  # Variable was deprecated
import numpy as np


def get_test_input():
    img = cv2.imread('dog-cycle-car.png')   # In BGR colorspace
    img = cv2.resize(img, (416, 416))
    img_ = img[:, :, ::-1].transpose((2, 0, 1))  # BGR -> RGB and HWC -> CHW
    # Add a new channel at dim 0 for batch and Normalisation
    img_ = img_[np.newaxis, :, :, :]/255.0
    img_ = torch.from_numpy(img_).float()   # numpy -> float tensor
    return img_


def parse_cfg(cfgfile):
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')                 # Lines ->  List
    lines = [x for x in lines if len(x) > 0]        # Remove empty lines
    lines = [x for x in lines if x[0] != '#']       # Remove comments
    lines = [x.rstrip().lstrip() for x in lines]    # Remove fringe whitespaces

    block = {}
    blocks = []
    for line in lines:
        if line[0] == "[":
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()     # Get the block name
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


def create_modules(blocks):
    """
    Construct PyTorch modules for the blocks.
    """
    net_info = blocks[0]    # Not a layer, but with info about the input/ preprocessing/ training params
    '''
    ModuleList is a sub-class of Module.
    '''
    module_list = nn.ModuleList()
    prev_filters = 3    # Keep track of the number of the filters in the last layer which is also the kernel depth in the current layer
    output_filters = []  # For route layers, keeping track of all output filter numbers

    # Iterate the blocks
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        '''
        1) Check the type of the block
        2) Create a new module for the block
        3) Append the module to `module_list`
        '''
        # Convolutional layer
        if x['type'] == 'convolutional':
            activation = x['activation']
            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x['filters'])
            padding = int(x['pad'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])

            if padding:
                pad = (kernel_size - 1)//2
            else:
                pad = 0

            # Add the conv layer
            conv = nn.Conv2d(in_channels=prev_filters, out_channels=filters,
                             kernel_size=kernel_size, stride=stride, padding=pad, bias=bias)
            module.add_module('conv_{}'.format(index), conv)

            # Add the Batch normalization layer
            if batch_normalize:
                bn = nn.BatchNorm2d(num_features=filters)
                module.add_module('batch_norm){0}'.format(index), bn)

            # Check the activation
            # Either Linear or Leaky ReLU for YOLO
            if activation == 'leaky':
                activn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
                module.add_module('leaky_{}'.format(index), activn)

        # Else: upsampling layer
        # Bilinear2dUpsampling
        elif x['type'] == 'upsample':
            stride = int(x['stride'])
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            module.add_module('upsample_{}'.format(index), upsample)

        # Else: route layer
        # Need to compute output depth (filters) due to concatenation
        elif x['type'] == 'route':
            x['layers'] = x['layers'].split(',')
            # start of a route
            start = int(x['layers'][0])
            # end, if there are two values in layers attribute
            try:
                end = int(x['layers'][1])
            except:
                end = 0
            # Positive annotation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module('route_{}'.format(index), route)

            if end < 0:
                filters = output_filters[index+start]+output_filters[index+end]
            else:
                filters = output_filters[index+start]

        # Else: shortcut layer
        # No need to compute output depth (filters) due to addition
        elif x['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module('shortcut_{}'.format(index), shortcut)

        # Else: yolo layer, detection layer
        elif x['type'] == 'yolo':
            mask = list(map(int, x['mask'].split(',')))

            anchors = list(map(int, x['anchors'].split(',')))
            anchors = [(anchors[i], anchors[i+1])
                       for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module('Detection_{}'.format(index), detection)

        # Appending works
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)


# EmptyLayer object to be concatenated in the forward()
# function of darknet obejct.
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


# YOLOv3 architecture
class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}    # Cache the outputs of all layers for the route/shortcut

        write = 0   # Indicate whether we have encountered the first detection layer
        for i, module in enumerate(modules):
            module_type = (module['type'])

            # convolutional and upsample layers: they have forward property
            if module_type == 'convolutional' or module_type == 'upsample':
                x = self.module_list[i](x)

            # route layer: concatenation from two output feature maps from other layers
            elif module_type == 'route':
                layers = module['layers']
                layers = [int(a) for a in layers]  # ',' Already split

                if layers[0] > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i+(layers[0])]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i
                    feature_map_1 = outputs[i+layers[0]]
                    feature_map_2 = outputs[i+layers[1]]

                    # Concatenation along depth
                    x = torch.cat((feature_map_1, feature_map_2), dim=1)

            # shortcut layer: addition of two output feature maps from other layers
            elif module_type == 'shortcut':
                from_layer = int(module['from'])
                # Addition
                x = outputs[i-1] + outputs[i+from_layer]

            # yolo layer/detection layer:
            elif module_type == 'yolo':

                anchors = self.module_list[i][0].anchors
                input_dim = int(self.net_info['height'])
                num_classes = int(module['classes'])

                x = predict_transform(
                    prediction=x,
                    input_dim=input_dim,
                    anchors=anchors,
                    num_classes=num_classes,
                    CUDA=CUDA
                )
                if not write:   # If no collector has been initialised
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x

        return detections


# Test the creation
if __name__ == '__main__':
    # Create the module list and print it
    # blocks = parse_cfg("cfg/yolov3.cfg")
    # # Check the architecture at https://i.loli.net/2020/04/04/T4yxQ3btrvhISog.png
    # print(create_modules(blocks))

    # Given an image, implement the forward pass once to get the prediction table
    model = Darknet('cfg/yolov3.cfg')
    inp = get_test_input()
    pred = model(inp, torch.cuda.is_available())
    print(pred.size())
    # print(pred)
