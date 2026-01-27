from models.resnet import *
import torch
import torch.nn as nn
import random


def random_crop_3d_tensor(tensor, crop_size=(96, 96, 64), num_crops=4):
    if tensor.dim() == 3:
        D, H, W = tensor.shape
        has_channel = False
    elif tensor.dim() == 4:  # 如果有 channel 维度 (C, D, H, W)
        C, D, H, W = tensor.shape
        has_channel = True
    else:
        raise ValueError("Input tensor must be 3D (D,H,W) or 4D (C,D,H,W)")

    c_d, c_h, c_w = crop_size

    if c_d > D or c_h > H or c_w > W:
        raise ValueError(f"Crop size {crop_size} is larger than input tensor size {(D, H, W)}")

    crops = []
    coords = []

    for _ in range(num_crops):
        # 随机生成起始坐标
        start_d = random.randint(0, D - c_d)
        start_h = random.randint(0, H - c_h)
        start_w = random.randint(0, W - c_w)

        end_d = start_d + c_d
        end_h = start_h + c_h
        end_w = start_w + c_w

        if has_channel:
            crop = tensor[:, start_d:end_d, start_h:end_h, start_w:end_w]
        else:
            crop = tensor[start_d:end_d, start_h:end_h, start_w:end_w]

        crops.append(crop.unsqueeze(0))
        coords.append((start_d, start_h, start_w))

    return crops, coords


def crop_3d_tensor(tensor, crop_size=(96, 96, 64), overlap_ratio=0.75):
    if tensor.dim() == 3:
        D, H, W = tensor.shape
    elif tensor.dim() == 4:
        C, D, H, W = tensor.shape
    else:
        raise ValueError("Input tensor must be 3D (D,H,W) or 4D (C,D,H,W)")

    c_d, c_h, c_w = crop_size

    overlap_d = int(c_d * overlap_ratio)
    overlap_h = int(c_h * overlap_ratio)
    overlap_w = int(c_w * overlap_ratio)

    stride_d = c_d - overlap_d
    stride_h = c_h - overlap_h
    stride_w = c_w - overlap_w

    n_d = (D - overlap_d + stride_d - 1) // stride_d
    n_h = (H - overlap_h + stride_h - 1) // stride_h
    n_w = (W - overlap_w + stride_w - 1) // stride_w

    crops = []
    coords = []
    for i in range(n_d):
        for j in range(n_h):
            for k in range(n_w):
                start_d = i * stride_d
                start_h = j * stride_h
                start_w = k * stride_w

                end_d = start_d + c_d
                end_h = start_h + c_h
                end_w = start_w + c_w

                if end_d > D:
                    start_d = D - c_d
                    end_d = D
                if end_h > H:
                    start_h = H - c_h
                    end_h = H
                if end_w > W:
                    start_w = W - c_w
                    end_w = W

                if tensor.dim() == 3:
                    crop = tensor[start_d:end_d, start_h:end_h, start_w:end_w]
                else:  # dim=4, (C,D,H,W)
                    crop = tensor[:, start_d:end_d, start_h:end_h, start_w:end_w]

                crops.append(crop.unsqueeze(0))

                coord = (start_d, start_h, start_w)
                coords.append(coord)

    return crops, coords


def crop_labels_with_coords(labels, coords, crop_size=(96, 96, 64),):
    c_d, c_h, c_w = crop_size
    crop_labels = []

    for coord in coords:

        start_d, start_h, start_w = coord
        end_d = start_d + c_d
        end_h = start_h + c_h
        end_w = start_w + c_w

        if labels.dim() == 3:
            crop = labels[start_d:end_d, start_h:end_h, start_w:end_w]
        else:  # dim=4 (C,D,H,W)
            crop = labels[:, start_d:end_d, start_h:end_h, start_w:end_w]

        crop_labels.append(crop.unsqueeze(0))

    crop_labels = torch.concat(crop_labels, dim=0)

    return crop_labels


class Glance_model(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, backbone='resnet18'):
        super(Glance_model, self).__init__()
        if backbone == 'resnet10':
            self.backbone = resnet10(pretrained=False, n_input_channels=1, num_classes=2, feed_forward=False,
                                     shortcut_type='B', bias_downsample=False)

        else:
            self.backbone = resnet18(pretrained=False, n_input_channels=1, num_classes=2, feed_forward=False,
                                     shortcut_type='A', bias_downsample=True)

        self.layer = nn.Sequential(
            nn.Linear(512, 2),
        )

    def get_actions(self, x):
        # inference with batch, to avoid large batch --> out of memory
        infer_batch = 16  # max batch
        total_samples = x.size()[0]
        results = []
        for i in range(0, total_samples, infer_batch):
            batch = x[i:i + infer_batch]
            output = self.backbone(batch)
            output = self.layer(output)
            results.append(output)

        # Concatenate all results
        actions = torch.cat(results, dim=0)

        return actions

    def forward(self, input_images):
        # for validation

        b = input_images.size()[0]
        xs = []
        coords_all = []

        for i in range(b):
            input_image = input_images[i][0]
            crops, coords = crop_3d_tensor(input_image)
            xs += crops
            coords_all.append(coords)
        # coords_all length: batch_size. each length: num of crops

        crops = torch.concat(xs, dim=0).unsqueeze(1)
        actions = self.get_actions(crops)

        return crops, coords_all, actions

    def forward_with_labels(self, input_images, input_labels, training=True):
        # for training with labels
        b = input_images.size()[0]
        xs = []
        coords_all = []

        for i in range(b):
            input_image = input_images[i][0]

            if training:
                crops, coords = random_crop_3d_tensor(input_image, num_crops=1)
            else:
                crops, coords = crop_3d_tensor(input_image)

            xs += crops
            coords_all.append(coords)
        # coords_all length: batch_size. each length: num of crops

        crops = torch.concat(xs, dim=0).unsqueeze(1)
        actions = self.get_actions(crops)

        labels_all = []
        # crop labels:
        for i in range(b):
            label = input_labels[i]
            coords_each = coords_all[i]
            crop_labels = crop_labels_with_coords(label, coords_each)
            labels_all.append(crop_labels)
        output_labels = torch.concat(labels_all, dim=0)

        return crops, coords_all, output_labels, actions