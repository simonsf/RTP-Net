from __future__ import print_function
import os
import numpy as np
import configparser
from easydict import EasyDict as edict
import torch.nn as nn
from utils.tools import *
from utils.Normalizer import *
    
    
class VSegModel(object):

    def __init__(self):
        self.net = None
        # image spacing
        self.spacing = np.array([1, 1, 1])
        # maximum stride of network
        self.max_stride = 1
        # image crop normalizers
        self.crop_normalizers = None
        # network outputs
        self.out_channels = 2
        # network input channels
        self.in_channels = 1
        # interpolation method
        self.interpolation = 'NN'
        # default paddding value list
        self.default_values = np.array([0], dtype=np.double)
        # name of network type
        self.network_type = None
        # sample method
        self.cropping_method = None
        # crop voxel size
        self.crop_voxel_size = np.array([0, 0, 0], dtype=np.int)
        # box percent padding
        self.box_percent_padding = 0.0
        # testing level
        self.level = None

    def load(self, model_dir):
        """ load python segmentation model from folder
        :param model_dir: model directory
        :return: None
        """
        if not os.path.isdir(model_dir):
            raise ValueError('model dir not found: {}'.format(model_dir))

        checkpoint_dir = last_checkpoint(os.path.join(model_dir, 'checkpoints'))
        param_file = os.path.join(checkpoint_dir, 'params.pth')

        if not os.path.isfile(param_file):
            raise ValueError('param file not found: {}'.format(param_file))

        # load network parameters
        state = load_pytorch_model(param_file)
        self.spacing = np.array(state['spacing'], dtype=np.double)
        assert self.spacing.ndim == 1, 'spacing must be 3-dim array'

        self.max_stride = state['max_stride']

        self.crop_normalizers = []
        
        if 'crop_normalizers' in state:
            for crop_normalizer in state['crop_normalizers']:
                self.crop_normalizers.append(self.__normalizer_from_dict(crop_normalizer))
        elif 'crop_normalizer' in state:
            self.crop_normalizers.append(self.__normalizer_from_dict(state['crop_normalizer']))
        else:
            raise ValueError('crop_normalizers not found in checkpoint')

        if 'default_values' in state:
            self.default_values = np.array(state['default_values'], dtype=np.double)
        else:
            self.default_values = np.array([0], dtype=np.double)

        self.out_channels = 2
        if 'out_channels' in state:
            self.out_channels = state['out_channels']

        self.in_channels = 1
        if 'in_channels' in state:
            self.in_channels = state['in_channels']

        self.interpolation = 'NN'
        if 'interpolation' in state:
            assert self.interpolation in ('NN', 'LINEAR', 'FILTER_NN'), '[Model] Invalid Interpolation'
            self.interpolation = state['interpolation']

        if 'cropping_method' in state:
            self.cropping_method = state['cropping_method']
            if self.cropping_method == 'fixed_box':
                self.crop_voxel_size = state['crop_voxel_size']
                self.box_percent_padding = state['box_percent_padding']
        else:
            self.cropping_method = 'fixed_spacing'
        assert self.cropping_method in ['fixed_spacing', 'fixed_box'], 'invalid cropping method'

        net_name = state['net']
        self.network_type = net_name

        if net_name == 'vbnet':
            from network import vbnet
            net_module = vbnet
            self.net = net_module.SegmentationNet(self.in_channels, self.out_channels)
        elif net_name == 'vbbnet':
            from network import vbbnet
            net_module = vbbnet
            self.net = net_module.SegmentationNet(self.in_channels, self.out_channels)
        else:
            raise ValueError("Net name should be either '2D_net' or '25D_net'!")
        self.net = nn.parallel.DataParallel(self.net)
        self.net = self.net.cuda()
        self.net.load_state_dict(state['state_dict'])
        self.net.eval()

    @staticmethod
    def __normalizer_from_dict(crop_normalizer):
        """ convert dictionary to crop normalizer """

        if crop_normalizer['type'] == 0:
            ret = Fix_normalizer(crop_normalizer['mean'], crop_normalizer['stddev'], crop_normalizer['clip'])
        elif crop_normalizer['type'] == 1:
            ret = Adaptive_normalizer(crop_normalizer['min_p'], crop_normalizer['max_p'], crop_normalizer['clip'])
        else:
            raise ValueError('unknown normalizer type: {}'.format(crop_normalizer['type']))
        return ret