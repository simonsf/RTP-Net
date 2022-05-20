from __future__ import print_function
import os
import numpy as np
import time
import argparse
import os
import numpy as np
import importlib
import codecs
from easydict import EasyDict as edict
import torch
import torch.nn as nn
from utils.tools import read_test_txt, read_test_csv, read_test_folder
from utils.dataset import fix_normalizers, adaptive_normalizers, resize_image_itk
import SimpleITK as sitk
import copy


def prepare_image_fixed_spacing(images, model):
    ori_spacing = images[0].GetSpacing()
    spacing = model.spacing
   
    prev_size = images[0].GetSize()

    box_size = (np.array(images[0].GetSize()) * ori_spacing / spacing + 0.5).astype(np.int32)

    for i in range(3):
        box_size[i] = int(np.ceil(box_size[i] * 1.0 / model.max_stride[i]) * model.max_stride[i])

    method = model.interpolation

    assert method in ('NN', 'LINEAR')

    resample_images = []
    iso_images = []
    for idx, image in enumerate(images):
        ret, params = model.crop_normalizers[idx]
        params['image'] = sitk.GetArrayFromImage(image)
        norm_data = ret(**params)

        image_origin =  image.GetOrigin()
        image_spacing = image.GetSpacing()
        image_direction = image.GetDirection()

        image = sitk.GetImageFromArray(norm_data)
        image.SetOrigin(image_origin)
        image.SetSpacing(image_spacing)
        image.SetDirection(image_direction)

        if method == 'NN':
            resample_image = resize_image_itk(image, box_size.tolist(), spacing.tolist())
        elif method == 'LINEAR':
            resample_image = resize_image_itk(image, box_size.tolist(), spacing.tolist(), resamplemethod=sitk.sitkLinear)
        resample_images.append(resample_image)

        resample_data = sitk.GetArrayFromImage(resample_image)
        iso_images.append(resample_data)      
  
    iso_image_tensor = torch.from_numpy(np.array(iso_images)).unsqueeze(0)

    return iso_image_tensor, images[0], resample_images[0]



def network_output(iso_batch, model, pre_image, resample_image):
    probs=[]
    with torch.no_grad():
        prob = model(iso_batch)

    _, mask = prob.max(1)
    mask = mask.short()

    mask = np.array((mask.data.cpu()))
    prob_map = np.array((prob[:, 1].data.cpu()))

    ori_origin =  resample_image.GetOrigin()
    ori_spacing = resample_image.GetSpacing()
    ori_direction = resample_image.GetDirection()

    tar_size =  pre_image.GetSize()
    tar_spacing = pre_image.GetSpacing()

    mask = sitk.GetImageFromArray(mask)
    mask.SetOrigin(ori_origin)
    mask.SetSpacing(ori_spacing)
    mask.SetDirection(ori_direction)

    prob_map = sitk.GetImageFromArray(prob_map)
    prob_map.SetOrigin(ori_origin)
    prob_map.SetSpacing(ori_spacing)
    prob_map.SetDirection(ori_direction)

    pre_mask = resize_image_itk(mask, tar_size, tar_spacing)
    pre_prob_map = resize_image_itk(prob_map, tar_size, tar_spacing)

    return pre_mask, pre_prob_map


def test(input_path, model_path, output_folder, seg_name='seg.mha', gpu_id=0, save_image=True, save_single_prob=True):
    total_test_time = 0
    model = torch.load(model_path)

    suffix = ['.mhd', '.nii', '.hdr', '.nii.gz', '.mha', '.image3d']
    if os.path.isfile(input_path):
        # test image files in the text (single-modality) or csv (multi-modality) file
        if input_path.endswith('txt'):
            file_list, case_list = read_test_txt(input_path)
        elif input_path.endswith('csv'):
            file_list, case_list = read_test_csv(input_path)
        else:
            raise ValueError('image test_list must either be a txt file or a csv file')
    elif os.path.isdir(input_path):
        # test all image file in input folder (single-modality)
        file_list, case_list = read_test_folder(input_path)
    else:
        raise ValueError('Input path do not exist!')

    success_cases = 0
    model_name = os.path.basename(model_path)
    model_in_channels = model.__dict__[model_name].in_channels
    for i, file in enumerate(file_list):
        print('{}: {}'.format(i, file))

        begin = time.time()
        images = []
        for image_path in file:
            image = sitk.ReadImage(image_path, outputPixelType=sitk.sitkFloat32)
            images.append(image)
        read_time = time.time() - begin

        begin = time.time()

        iso_batch, pre_image, resample_image = prepare_image_fixed_spacing(images[:model_in_channels], model[model_name])
        mask, prob_map = network_output(iso_batch, model[model_name], pre_image, resample_image)
        test_time = time.time() - begin

        casename = case_list[i]
        out_folder = os.path.join(output_folder, casename)
        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)

        begin = time.time()
        if save_image:
            if len(images) == 1:
                ct_path = os.path.join(out_folder, 'org.nii.gz')
                sitk.WriteImage(images[0], ct_path)
            else:
                for num in range(len(images)):
                    ct_path = os.path.join(out_folder, 'org{}'.format(num+1) + '.nii.gz')
                    sitk.WriteImage(images[num], ct_path)

        seg_path = os.path.join(out_folder, seg_name)
        sitk.WriteImage(mask, seg_path)

        if save_single_prob and prob_map or True:
            prob_path = os.path.join(out_folder, 'prob.nii.gz')
            sitk.WriteImage(prob_map, prob_path)
        output_time = time.time() - begin

        total_time = read_time + test_time + output_time
        total_test_time = test_time + total_test_time
        success_cases += 1
        print('read: {:.2f} s, test: {:.2f} s, write: {:.2f} s, total: {:.2f} s, avg test time: {:.2f}'.format(
            read_time, test_time, output_time, total_time, total_test_time / float(success_cases)))


def main():

    from argparse import RawTextHelpFormatter

    long_description = 'UII RTP-Net Testing Engine\n\n' \
                       'It supports multiple kinds of input:\n' \
                       '1. Image list txt file\n' \
                       '2. Single image file\n' \
                       '3. A folder that contains all testing images\n'

    parser = argparse.ArgumentParser(description=long_description,
                                     formatter_class=RawTextHelpFormatter)

    parser.add_argument('-i', '--input', type=str, help='input folder/file for intensity images', default='/data/qingzhou/whm/vseg_test.csv')
    parser.add_argument('-m', '--model', type=str, help='pth model path', default='/data/qingzhou/whm/wmh_2d_4.pth')
    parser.add_argument('-o', '--output', type=str, help='output folder for segmentation', default='/data/qingzhou/whm/wmh_2d/predicts')
    parser.add_argument('-n', '--seg_name', default='seg2.nii.gz', help='the name of the segmentation result to be saved')
    parser.add_argument('-g', '--gpu_id', default='5', help='the gpu id to run model')
    parser.add_argument('--save_image', help='whether to save original image', action="store_true")
    parser.add_argument('--save_single_prob', help='whether to save single prob map', action="store_true")
    args = parser.parse_args()
    test(args.input, args.model, args.output, args.seg_name, int(args.gpu_id), args.save_image,
                 args.save_single_prob)


if __name__ == '__main__':
    main()


