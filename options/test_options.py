#! /usr/bin/python3
# -*- coding:utf-8 -*-
# @Time  : 18-10-30 下午11:47
# Author : TJJ
from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        parser.set_defaults(model='test')
        self.isTrain = False
        return parser