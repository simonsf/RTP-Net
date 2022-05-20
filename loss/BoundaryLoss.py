import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sigmoid_weight(k, dist):
    return 2 / (1 + torch.exp(k * dist))


def generate_pooling(level=20):
    max_pool = []
    for k in range(0, level+1):
        max_pool.append(torch.nn.MaxPool3d([1, 2 * k + 1, 2 * k + 1], 1, [0, k, k]).cuda())
    return max_pool


def generate_pooling_2d(level=20):
    max_pool = []
    for k in range(0, level+1):
        max_pool.append(torch.nn.MaxPool2d([2 * k + 1, 2 * k + 1], 1, [k, k]).cuda())
    return max_pool


def cal_dist(mask, max_pool, dim):
    filt = -1 * np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                     [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                     [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    filt = torch.Tensor(filt)
    filt = filt.unsqueeze(0).unsqueeze(0) / 4
    filt = filt.cuda()
    filt_2d = filt[:, :, 1, :, :]

    level = len(max_pool) + 1
    mask = mask.float()
    if dim == 3:
        edge = (F.conv3d(mask, filt, padding=1) > 0).float()
    else:
        edge = (F.conv2d(mask, filt_2d, padding=1) > 0).float()
    dist = torch.zeros_like(edge).float().cuda()
    for mpk in max_pool:
        dist += mpk(edge)
    dist = (level - dist) / level
    return dist


class BinaryBoundarySoftDice(nn.Module):
    '''
    requires 4D or 5D input for dist calculating
    '''
    def __init__(self, k, level=20, dim=3):
        super(BinaryBoundarySoftDice, self).__init__()
        if dim == 3:
            self.max_pool = generate_pooling(level=level)
        else:
            self.max_pool = generate_pooling_2d(level=level)
        self.dim = dim
        self.k = k

    def forward(self, outputs, masks, dist=None):
        epsilon = 1e-6
        if dist is None:
            dist = cal_dist(masks, self.max_pool, self.dim)
        assert dist.shape == masks.shape
        self.weight = sigmoid_weight(self.k, dist.float()).detach()

        batchsize = outputs.size(0)
        #outputs_1 = outputs.index_select(1, torch.arange(0, 1).cuda()).float()
        outputs_1 = outputs.float()
        masks = masks.float()
        outputs_w = outputs_1 * self.weight
        masks_w = masks * self.weight
        outputs_w = outputs_w.view(batchsize, -1)
        masks_w = masks_w.view(batchsize, -1)

        intersect = torch.sum(outputs_w * masks.view(batchsize, -1), 1)
        input_area = torch.sum(outputs_w, 1)
        target_area = torch.sum(masks_w, 1)
        sum = input_area + target_area + 2 * epsilon

        batch_loss = 1 - 2 * intersect / sum
        batch_loss[target_area == 0] = 0
        loss = batch_loss.mean()

        return loss


class BoundarySoftDice(nn.Module):
    def __init__(self, k, weights, num_class, level=20, dim=3):
        super(BoundarySoftDice, self).__init__()
        self.max_pool = generate_pooling(level=level)
        self.k = k
        self.level = level
        self.num_class = num_class
        self.weights = torch.FloatTensor(weights)
        self.weights = self.weights / self.weights.sum()
        self.weights = self.weights.cuda()
        self.dim = dim
    
    def set_k(self, k):
        self.k = k

    def forward(self, outputs, masks, dist=None):
        dice_losses = []
        weight_dice_loss = 0
        all_slice = torch.split(outputs, [1] * self.num_class, dim=1)
        for i in range(self.num_class):
            # prepare for calculate label i dice loss
            slice_i = all_slice[i]
            target_i = (masks == i) * 1

            BBD = BinaryBoundarySoftDice(self.k, self.level, self.dim)
            dice_i_loss = BBD(slice_i, target_i, dist=dist)

            dice_losses.append(dice_i_loss)
            weight_dice_loss += dice_i_loss * self.weights[i]

        return weight_dice_loss, [dice_loss.item() for dice_loss in dice_losses]
