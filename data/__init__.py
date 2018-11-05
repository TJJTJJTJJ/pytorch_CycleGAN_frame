#! /usr/bin/python3
# -*- coding:utf-8 -*-
# @Time  : 18-10-30 下午10:43
# Author : TJJ

__all__ = ['CreateDataLoader', 'get_option_setter']
"""
重点：CustomDatasetDataLoader
"""
#################################
"""
train.py
CreateDataLoader----CustomDatasetDataLoader.initialize----BaseDataLoader.initialize
                                                     |----create_dataset----find_dataset_using_name
                                                                       |----AlignedDataset.initialize
base_options.py                                                                       
get_option_setter----AlignedDataset.modify_commandline_options                                                                  
"""
#################################
"""
可以发现，这个文件内是无视具体的dataset的，抽象性极高，可任意迁移
__init__.py    base_data_loader.py  base_dataset.py  image_folder.py
"""
def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(opt)
    return data_loader


# Wrapper class of Dataset class that performs
# multi-threaded data loading
class CustomDatasetDataLoader(BaseDataLoader):
    """
    无视具体的具体的数据集而生成dataloader
    """
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = create_dataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data

def create_dataset(opt):
    """
    根据opt生成data类
    :param opt:  opt.dataset_mode
    :return: class AlignedDataset
    """
    dataset = find_dataset_using_name(opt.dataset_mode)
    instance = dataset()
    instance.initialize(opt)
    print("dataset [%s] was created" % (instance.name()))
    return instance

def find_dataset_using_name(dataset_name):
    # Given the option --dataset_mode [datasetname],
    # the file "data/datasetname_dataset.py"
    # will be imported.
    """

    :param dataset_name: unaligned
    :return: dataset_class AlignedDataset
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    # In the file, the class called DatasetNameDataset() will
    # be instantiated. It has to be a subclass of BaseDataset,
    # and it is case-insensitive.
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        print("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))
        exit(0)

    return dataset


def get_option_setter(dataset_name):
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options