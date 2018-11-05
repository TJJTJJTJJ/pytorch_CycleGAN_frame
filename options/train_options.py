#! /usr/bin/python3
# -*- coding:utf-8 -*-
# @Time  : 18-10-30 下午11:46
# Author : TJJ
from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        self.isTrain = True
        return parser