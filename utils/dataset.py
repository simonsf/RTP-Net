# -*- coding:utf-8 -*-
import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import codecs
import SimpleITK as sitk
from utils.tools import *
from utils.Normalizer import *

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

def read_train_txt(imlist_file):
    """ read single-modality txt file
    :param imlist_file: image list file path
    :return: a list of image path list, list of segmentation paths
    """
    lines = readlines(imlist_file)
    num_cases = int(lines[0])

    if len(lines)-1 < num_cases * 2:
        raise ValueError('too few lines in imlist file')

    im_list, seg_list = [], []
    for i in range(num_cases):
        im_path, seg_path = lines[1 + i * 2], lines[2 + i * 2]
        assert os.path.isfile(im_path), 'image not exist: {}'.format(im_path)
        assert os.path.isfile(seg_path), 'mask not exist: {}'.format(seg_path)
        im_list.append([im_path])
        seg_list.append(seg_path)

    return im_list, seg_list


def read_train_csv(csv_file):
    """ read multi-modality csv file
    :param csv_file: csv file path
    :return: a list of image path list, list of segmentation paths
    """
    im_list, seg_list = [], []
    with open(csv_file, 'r') as fp:
        reader = csv.reader(fp)
        headers = next(reader)
        num_modality = len(headers) - 1
        for i in range(num_modality):
            assert headers[i] == 'image{}'.format(i+1)
        assert headers[-1] == 'segmentation'

        for line in reader:
            for path in line:
                assert os.path.isfile(path), 'file not exist: {}'.format(path)
            im_list.append(line[:-1])
            seg_list.append(line[-1])
    return im_list, seg_list


def resize_image_itk(ori_img, target_Size, target_Spacing, resamplemethod=sitk.sitkNearestNeighbor, pixel_type=sitk.sitkFloat32):
    # target_Size = target_img.GetSize()      # 目标图像大小  [x,y,z]
    # target_Spacing = target_img.GetSpacing()   # 目标的体素块尺寸    [x,y,z]
    target_origin = ori_img.GetOrigin()      # 目标的起�?[x,y,z]
    target_direction = ori_img.GetDirection()  # 目标的方�?[�?�?横]=[z,y,x]


    # itk的方法进行resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ori_img)  # 需要重新采样的目标图像
    # 设置目标图像的信�?    
    resampler.SetSize(target_Size)		# 目标图像大小
    resampler.SetOutputOrigin(target_origin)
    resampler.SetOutputDirection(target_direction)
    resampler.SetOutputSpacing(target_Spacing)
    # 根据需要重采样图像的情况设置不同的dype
    if resamplemethod == sitk.sitkNearestNeighbor:
        # resampler.SetOutputPixelType(sitk.sitkUInt16)   # 近邻插值用于mask的，保存uint16
        resampler.SetOutputPixelType(pixel_type)
    else:
        resampler.SetOutputPixelType(sitk.sitkFloat32)  # 线性插值用于PET/CT/MRI之类的，保存float32
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))    
    resampler.SetInterpolator(resamplemethod)
    itk_img_resampled = resampler.Execute(ori_img)  # 得到重新采样后的图像

    return itk_img_resampled

def center_crop(image, coord, size, padtype=0, padvalue=0):
    """
    crop a sub-volume centered at voxel.
    :param image:       an image3d object
    :param coord:       the coordinate of center voxel
    :param spacing:     spacing of output volume
    :param size:        size of output volume
    :param padtype:     padding type, 0 for value padding and 1 for edge padding
    :param padvalue:    the default padding value for value padding
    :return: a sub-volume image3d object
    """

    assert padtype in [0, 1], 'padtype not support'
    coord = np.array(coord, dtype=np.int32)
    data = sitk.GetArrayFromImage(image)
    #slice_data = data[coord[2],:,:]
    data_size = data.shape
    new_center = np.array([coord[2],coord[1],coord[0]], dtype=np.int32)
    if padtype == 0:
        pad_mode = 'constant'
    else:
        pad_mode = 'edge'
        

    pad_data = np.pad(data, 
                      ((max(0, int(size[0]//2-new_center[0])), max(int(size[0]//2+size[0]%2+new_center[0]-data_size[0]), 0)),
                       (max(0, int(size[1]//2-new_center[1])), max(int(size[1]//2+size[1]%2+new_center[1]-data_size[1]), 0)),
                       (max(0, int(size[2]//2-new_center[2])), max(int(size[2]//2+size[2]%2+new_center[2]-data_size[2]), 0))
                       
                      ),
                      mode = pad_mode,
                      constant_values = (padvalue, padvalue)
                      )
    new_center_pad = [new_center[0]+max(0, int(size[0]//2-new_center[0])), new_center[1]+max(0, int(size[1]//2-new_center[1])), new_center[2]+max(0, int(size[2]//2-new_center[2]))]
    
    crop_data = pad_data[-size[0]//2+new_center_pad[0]:size[0]//2+size[0]%2+new_center_pad[0],
                         -size[1]//2+new_center_pad[1]:size[1]//2+size[1]%2+new_center_pad[1],
                         -size[2]//2+new_center_pad[2]:size[2]//2+size[2]%2+new_center_pad[2]]

    return crop_data


class SegmentationDataset(Dataset):
    """ training data set for volumetric segmentation """

    def __init__(self, imlist_file, num_classes, spacing, crop_size, default_values, sampling_method,
                 random_translation,
                 interpolation, crop_normalizers,
                 rai_sample=False
                ):
        """ constructor
        :param imlist_file: image-segmentation list file
        :param num_classes: the number of classes
        :param spacing: the resolution, e.g., [1, 1, 1]
        :param crop_size: crop size, e.g., [96, 96, 96]
        :param default_values: default padding value list, e.g.,[0]
        :param sampling_method: 'GLOBAL', 'MASK'
        :param random_translation: random translation
        :param interpolation: 'LINEAR' for linear interpolation, 'NN' for nearest neighbor
        :param crop_normalizers: used to normalize the image crops, one for one image modality
        """
        if imlist_file.endswith('txt'):
            self.im_list, self.seg_list = read_train_txt(imlist_file)
        elif imlist_file.endswith('csv'):
            self.im_list, self.seg_list = read_train_csv(imlist_file)
        else:
            raise ValueError('imseg_list must either be a txt file or a csv file')

        self.num_classes = num_classes
        self.default_values = default_values

        self.spacing = np.array(spacing, dtype=np.double)

        self.crop_size = np.array(crop_size, dtype=np.int32)

        assert sampling_method in ('GLOBAL', 'MASK', 'CENTER', 'MIX', 'RANDOM'), 'sampling_method must either be GLOBAL or MASK or CENTER or MIX or RANDOM'
        self.sampling_method = sampling_method

        self.random_translation = np.array(random_translation, dtype=np.double)
        assert self.random_translation.size == 3, 'Only 3-element of random translation is supported'

        assert interpolation in ('LINEAR', 'NN'), 'interpolation must either be a LINEAR or NN'
        self.interpolation = interpolation

        assert isinstance(crop_normalizers, list), 'crop normalizers must be a list'
        self.crop_normalizers = crop_normalizers

        self.rai_sample = rai_sample

        #self.aug = iaa.GammaContrast((0.6, 1.5))
        #self.aug_blur = iaa.AverageBlur(k=((1, 6), 1))
        #self.aug_he = iaa.AllChannelsCLAHE(clip_limit=(1, 10), per_channel=True)
        #self.aug_noise = iaa.AdditiveGaussianNoise(scale=(0.001, 0.04), per_channel=True)

    def __len__(self):
        """ get the number of images in this data set """
        return len(self.im_list)

    def num_modality(self):
        """ get the number of input image modalities """
        return len(self.im_list[0])

    def mask_sample(self, seg):
        data = sitk.GetArrayFromImage(seg)
        center_list = (data!=0).nonzero()
        if len(center_list[0]) > 0:
            idx = random.choice(range(len(center_list[0])))
            center = [center_list[2][idx], center_list[1][idx], center_list[0][idx]]
        else:
            center = self.global_sample(seg)
        return center

    def global_sample(self, seg):
        im_size = np.array(seg.GetSize())
        sp = np.array(im_size, dtype=np.double)
        for i in range(3):
            if im_size[i] > self.crop_size[i]:
                sp[i] = np.random.uniform(self.crop_size[i] / 2, im_size[i] - self.crop_size[i] / 2)
            else:
                sp[i] = np.random.uniform(0, im_size[i])
        center = np.rint(sp)
        return center

    def center_sample(self, seg):
        im_size = np.array(seg.GetSize())
        return np.rint(im_size / 2)

    def generate_center(self, seg, method):
        seed = np.random.choice([0,1,2,3],size=1,p=[0.25,0.25,0.25,0.25])[0]
        if method == 'MASK' or (method == 'RANDOM' and seed==0):
            center = self.mask_sample(seg)
            return center
        if method == 'GLOBAL' or (method == 'RANDOM' and seed==1):
            center = self.global_sample(seg)
            return center
        if method == 'CENTER' or (method == 'RANDOM' and seed==2):
            center = self.center_sample(seg)
            return center
        if method == 'MIX' or (method == 'RANDOM' and seed==3):
            slice_center = self.mask_sample(seg)[2]
            center = self.global_sample(seg)[:2]
            center = np.append(center, slice_center)
            return center            

    def __getitem__(self, index):

        image_paths, seg_path = self.im_list[index], self.seg_list[index]

        case_name = os.path.basename(os.path.dirname(image_paths[0]))
        case_name += '_' + os.path.basename(image_paths[0])
        images = []
        for image_path in image_paths:
            image = sitk.ReadImage(image_path, outputPixelType=sitk.sitkFloat32)
            images.append(image)

        seg = sitk.ReadImage(seg_path, outputPixelType=sitk.sitkFloat32)

        sampling_method = self.sampling_method
               
        ori_origin =  images[0].GetOrigin()
        ori_spacing = images[0].GetSpacing()
        ori_direction = images[0].GetDirection()

        spacing = self.spacing
        output_shape = (np.array(seg.GetSize()) * ori_spacing / spacing + 0.5).astype(np.int32)

        for idx in range(len(images)):
            if self.crop_normalizers[idx] is not None:
                if self.crop_normalizers[idx]['modality'] == 'CT':
                    normalizer = Fix_normalizer(float(self.crop_normalizers[idx]['mean']), float(self.crop_normalizers[idx]['stddev']), self.crop_normalizers[idx]['clip'])
                    norm_data = normalizer(sitk.GetArrayFromImage(images[idx]))
                    image = sitk.GetImageFromArray(norm_data)
                    image.SetOrigin(ori_origin)
                    image.SetSpacing(ori_spacing)
                    image.SetDirection(ori_direction)
                elif self.crop_normalizers[idx]['modality'] == 'MR':
                    normalizer = Adaptive_normalizer(float(self.crop_normalizers[idx]['min_p']), float(self.crop_normalizers[idx]['max_p']), self.crop_normalizers[idx]['clip'])
                    norm_data = normalizer(sitk.GetArrayFromImage(images[idx]))         
                    image = sitk.GetImageFromArray(norm_data)
                    image.SetOrigin(ori_origin)
                    image.SetSpacing(ori_spacing)
                    image.SetDirection(ori_direction)
            images[idx] = image
                      
        for idx in range(len(images)):
            if self.interpolation == 'NN':
                image = resize_image_itk(images[idx], output_shape.tolist(), spacing.tolist())
            elif self.interpolation == 'LINEAR':
                image = resize_image_itk(images[idx], output_shape.tolist(), spacing.tolist(), resamplemethod=sitk.sitkLinear)
            images[idx] = image
        
        seg = resize_image_itk(seg, output_shape.tolist(), spacing.tolist())
        
        center = self.generate_center(seg, sampling_method)
        voxel_translation = self.random_translation / ori_spacing[:3]
        trans = np.random.uniform(-voxel_translation, voxel_translation, size=[3]).astype(np.int16)
        #trans = np.append(trans, 0)
        center += trans
        #center = seg.world_to_voxel(center).astype(np.int16)

        for idx in range(len(images)):
            images[idx] = center_crop(images[idx], center, self.crop_size, padvalue=self.default_values[idx])

        seg = center_crop(seg, center, self.crop_size, padvalue=0)   

        axis = random.choice([0, 1, 2, 3, 4, 5])
        if axis in [0, 1, 2]:
            for idx in range(len(images)):
                images[idx] = np.flip(images[idx], axis)

            seg = np.flip(seg, axis)

        # convert to tensors
        im = torch.from_numpy(np.array(images))
        seg = torch.from_numpy(np.array([seg]))

        return im, seg, case_name




