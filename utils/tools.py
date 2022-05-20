import io
import torch
from Crypto.Cipher import AES
import base64
import random
import codecs
import os
import glob
import csv
from torch.utils.data.sampler import Sampler


class EpochConcateSampler(Sampler):
    """Concatenate  all epoch index arrays into one index array.

    Arguments:
        data_source (Dataset): dataset to sample from
        epoch(int): epoch num
    """

    def __init__(self, data_source, epoch):
        self.data_length = len(data_source)
        self.epoch = epoch

    def __iter__(self):
        index_all = []
        for i in range(self.epoch):
            index = list(range(self.data_length))
            random.shuffle(index)
            index_all += index
        return iter(index_all)

    def __len__(self):
        return self.data_length * self.epoch



def last_checkpoint(chk_root):
    """
    find the directory of last check point
    :param chk_root: the check point root directory, which may contain multiple checkpoints
    :return: the last check point directory
    """

    last_epoch = -1
    chk_folders = os.path.join(chk_root, 'chk_*')
    for folder in glob.glob(chk_folders):
        folder_name = os.path.basename(folder)
        tokens = folder_name.split('_')
        epoch = int(tokens[-1])
        if epoch > last_epoch:
            last_epoch = epoch

    if last_epoch == -1:
        raise OSError('No checkpoint folder found!')

    return os.path.join(chk_root, 'chk_{}'.format(last_epoch))


def readlines(file):
    """
    read lines by removing '\n' in the end of line
    :param file: a text file
    :return: a list of line strings
    """
    fp = codecs.open(file, 'r', encoding='utf-8')
    linelist = fp.readlines()
    fp.close()
    for i in range(len(linelist)):
        linelist[i] = linelist[i].rstrip('\n')
    return linelist


def read_test_txt(txt_file):
    """ read single-modality txt file
    :param txt_file: image list txt file path
    :return: a list of image path list, list of image case names
    """
    lines = readlines(txt_file)
    case_num = int(lines[0])

    if len(lines) - 1 < case_num:
        raise ValueError('case num cannot be greater than path num!')

    file_list, name_list = [], []
    for i in range(case_num):
        im_msg = lines[1 + i]
        im_msg = im_msg.strip().split()
        im_name = im_msg[0]
        im_path = im_msg[1]
        if not os.path.exists(im_path):
            raise ValueError('image not exist: {}'.format(im_path))
        file_list.append([im_path])
        name_list.append(im_name)

    return file_list, name_list


def read_test_csv(csv_file):
    """ read multi-modality csv file
    :param csv_file: image list csv file path
    :return: a list of image path list, list of image case names
    """
    file_list, name_list = [], []
    with open(csv_file, 'r') as fp:
        reader = csv.reader(fp)
        headers = next(reader)
        num_modality = len(headers) - 1
        for i in range(1, num_modality):
            assert headers[i] == 'image{}'.format(i)
        assert headers[0] == 'case_name'

        for line in reader:
            for path in line[1:]:
                assert os.path.exists(path), 'file not exist: {}'.format(path)
            file_list.append(line[1:])
            name_list.append(line[0])

    return file_list, name_list


def read_test_folder(folder_path):
    """ read single-modality input folder
    :param folder_path: image file folder path
    :return: a list of image path list, list of image case names
    """
    suffix = ['.mhd', '.nii', '.hdr', '.nii.gz', '.mha', '.image3d']
    file = []
    for suf in suffix:
        file += glob.glob(os.path.join(folder_path, '*' + suf))

    file_list, name_list = [], []
    for im_pth in sorted(file):
        _, im_name = os.path.split(im_pth)
        for suf in suffix:
            idx = im_name.find(suf)
            if idx != -1:
                im_name = im_name[:idx]
                break
        file_list.append([im_pth])
        name_list.append(im_name)

    return file_list, name_list