import io
import torch
import numpy as np
from Crypto.Cipher import AES
import base64
import random
import codecs
import os
import glob
import csv
import easydict
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


class Crypto(object):
    """Crypto provide bytes encrypt and decrypt function which mixes AES and base64."""
    def __init__(self, key=None):
        """
        :param key: password
        """
        if key is None:
            key = "*c!q9Kj*k?2>+5@p"
        assert len(key) == 16
        self.key = key
        self.mode = AES.MODE_CFB

    def bytes_encrypt(self, plain_text):
        """
        :param plain_text:
        :return: cipher_text(bytes)
        """
        assert isinstance(plain_text, bytes)

        length = 16
        plain_text = plain_text + b'\1'
        count = len(plain_text)
        add = length - (count % length)
        plain_text = plain_text + (b'\0' * add)

        aes_handle = AES.new(self.key, self.mode, self.key)
        cipher_text = base64.b64encode(aes_handle.encrypt(plain_text))

        return cipher_text

    def bytes_decrypt(self, cipher_text):
        """
        :param cipher_text:
        :return: plaintext(bytes)
        """
        assert isinstance(cipher_text, bytes)

        aes_handle = AES.new(self.key, self.mode, self.key)
        plain_text = aes_handle.decrypt(base64.b64decode(cipher_text))
        
        return plain_text.rstrip(b'\0')[0:-1]
        

def load_pytorch_model(path):
    """
    :param path: model path
    :return: model params
    """
    with open(path, "rb") as fid:
        buffer = io.BytesIO(fid.read())
        buffer_value = buffer.getvalue()

        if buffer_value[0:9] == b"uAI_model":
            crypto_handle = Crypto()
            decrypt_buffer = io.BytesIO(crypto_handle.bytes_decrypt(buffer_value[128::]))
        else:
            decrypt_buffer = buffer
    params = torch.load(decrypt_buffer)
    return params
      
    
def save_pytorch_model(params, save_path, is_encrypt=True):
    """
    :param params: model params
    :param save_path: model save path
    :param is_encrypt: encrypt or not
    :return: None
    """
    if not is_encrypt:
        torch.save(params, save_path)
        return

    buffer = io.BytesIO()
    torch.save(params, buffer)
    tag = b"uAI_model"
    tag = tag + b'\0'*(128 - len(tag))

    crypto_handle = Crypto()
    encrypt_buffer = tag + crypto_handle.bytes_encrypt(buffer.getvalue())

    with open(save_path, "wb") as fid:
        fid.write(encrypt_buffer)


def fix_normalizers(image, mean, stddev, clip=True):
    data = image
    if clip:
        return np.clip((data - mean) / (stddev + 1e-8) , -1, 1).astype(np.float32)
    else:
        return ((data - mean) / (stddev + 1e-8)).astype(np.float32)

def adaptive_normalizers(image, min_p, max_p, clip=True):
    data = image
    upper = np.percentile(data, max_p*100)
    lower = np.percentile(data, min_p*100)
    mean = (lower + upper) / 2.0
    stddev = abs((upper - lower)) / 2.0
    if clip:
        return np.clip((image - mean) / (stddev + 1e-8), -1, 1).astype(np.float32)
    else:
        return (image - mean) / (stddev + 1e-8).astype(np.float32)
        
        
def normalization_to_dict(crop_normalizers):
    assert type(crop_normalizers) == easydict.EasyDict
    norm_dict={}
    if crop_normalizers['modality'] == 'CT':
        norm_dict['type']=0
        norm_dict['mean']=float(crop_normalizers['mean'])
        norm_dict['stddev']=float(crop_normalizers['stddev'])
        if crop_normalizers.get('clip'):
            norm_dict['clip']=crop_normalizers['clip']
        else:
            norm_dict['clip']=True
    elif crop_normalizers['modality'] == 'MR':
        norm_dict['type']=1
        norm_dict['min_p']=float(crop_normalizers['min_p'])
        norm_dict['max_p']=float(crop_normalizers['max_p'])
        if crop_normalizers.get('clip'):
            norm_dict['clip']=crop_normalizers['clip']
        else:
            norm_dict['clip']=True
    return norm_dict
    
    