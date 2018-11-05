#! /usr/bin/python3
# -*- coding:utf-8 -*-
# @Time  : 18-10-30 下午11:28
# Author : TJJ
import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image

__all__ = ['AlignedDataset']


class AlignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):


    def __getitem__(self, index):
        """

        :param index:
        :return: dict {}
        """
        return {'A': A, 'B': B,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'
