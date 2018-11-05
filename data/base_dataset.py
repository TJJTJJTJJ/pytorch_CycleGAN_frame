#! /usr/bin/python3
# -*- coding:utf-8 -*-
# @Time  : 18-10-30 下午11:16
# Author : TJJ
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
__all__ = ['BaseDataset', 'get_transform']

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        pass

    def __len__(self):
        return 0


def get_transform(opt):
    """
    可有可无，这个函数可以直接定义在dataset等等，这个函数不重要
    :param opt: opt.resize_or_crop  opt.isTrain  opt.no_flip
    :return: transforms
    """
    return transforms.Compose(transform_list)

