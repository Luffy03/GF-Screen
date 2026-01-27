from models.glance import Glance_model
from monai.networks.nets import SwinUNETR

import torch
import torch.nn as nn
import torch.nn.functional as F


class GFScreen_RL(nn.Module):
    def __init__(self, args):
        super(GFScreen_RL, self).__init__()
        self.glance_model = Glance_model(backbone=args.glance_backbone)

        self.segmentation_model = SwinUNETR(img_size=(args.roi_x, args.roi_y, args.roi_z),
                                            in_channels=args.in_channels,
                                            out_channels=args.out_channels,
                                            feature_size=args.feature_size,
                                            use_checkpoint=args.use_checkpoint,
                                            use_v2=True)

        self.num_crops = args.batch_size * args.sw_batch_size
        self.args = args

    def get_RL_loss(self, actions_logits, old_policy_model, seg_crops, seg, seg_labels):
        # GRPO reinforcement training for Glance and Focus
        # actions_logits: (b, 2) - original logits
        # seg_crops: (b, 1, h, w, d), selected input
        # seg: segmentation predictions, (b, 2, h, w, d)
        # seg_labels: (b, 1, h, w, d), 0 and 1

        # larger temperature, smooth logits
        temperature = 1.0
        actions_softmax = F.softmax(actions_logits / temperature, dim=1)[:, 1].unsqueeze(1)

        # log actions_softmax
        actions_log_prob = torch.log(actions_softmax + 1e-6)

        # get old actions from old policy model, no gradient
        with torch.no_grad():
            old_actions = old_policy_model.get_actions(seg_crops)
            old_actions_softmax = F.softmax(old_actions / temperature, dim=1)[:, 1].unsqueeze(1)
            old_actions_log_prob = torch.log(old_actions_softmax + 1e-6)
            old_actions_log_prob = old_actions_log_prob.detach()

            # Get reward
            reward = get_detection_reward(pred=seg.detach(), label=seg_labels)

            # Group advantage from rewards. advantages shape: (b, 1)
            advantages = group_relative_advantages(rewards=reward)
            advantages = advantages.unsqueeze(1)

        # actions_log_prob, old_actions_log_prob: (b, 1)
        ratio = torch.exp(actions_log_prob - old_actions_log_prob)
        ratio = torch.clamp(ratio, 0.01, 5)  # clamp abnormal value

        # GRPO clip
        epsilon = 0.1
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
        loss_clip = - torch.mean(torch.min(surr1, surr2))

        # GRPO KL divergence
        beta = 0.01
        p = old_actions_softmax / (actions_softmax + 1e-6)
        p = torch.clamp(p, 1e-4, 1e4)  # clamp abnormal value
        KL_divergence = p - torch.log(p + 1e-6) - 1
        KL_divergence = beta * KL_divergence.mean()

        RL_loss = loss_clip + KL_divergence

        return RL_loss

    def forward(self, input_images, input_labels, old_policy_model):
        b, _, x, y, z = input_images.size()

        # cropped images, labels, and actions
        # actions.shape: (crops, 2)
        crops, coords_all, labels, actions = self.glance_model.forward_with_labels(input_images, input_labels,
                                                                                   training=True)

        # random indices
        indices = torch.randperm(b)

        # actions for RL loss
        select_actions_logits = actions[indices][:self.num_crops]

        # segmentation
        seg_crops = crops[indices][:self.num_crops]
        seg_labels = labels[indices][:self.num_crops]
        seg = self.segmentation_model(seg_crops)

        # get glance loss
        labels = get_glance_label(labels)
        glance_loss = get_glance_loss(actions, labels.squeeze(1))

        # get RL loss
        RL_loss = self.get_RL_loss(actions_logits=select_actions_logits, old_policy_model=old_policy_model,
                                   seg_crops=seg_crops, seg=seg, seg_labels=seg_labels)

        return seg, seg_labels, actions, labels, glance_loss, RL_loss

    def valid_with_labels(self, input_images, input_labels):
        b, _, x, y, z = input_images.size()
        assert b == 1, print('validation, batch_size should be 1, current batch_size is: ', b)

        # actions.shape: (crops, 2)
        crops, coords_all, labels, actions = self.glance_model.forward_with_labels(input_images, input_labels, training=False)

        # select by actions
        actions_argmax = actions.argmax(1)

        if actions_argmax.sum() > 0:
            selected_index = (actions_argmax == 1)

            selected_crops = crops[selected_index]

            # Convert coord to a tensor first
            coord = torch.tensor(coords_all[0]).to(input_images.device)
            selected_coord = coord[selected_index]

            # inference with batch, to avoid large batch for segmentation --> out of memory
            infer_batch = 4
            total_samples = selected_crops.size()[0]
            seg = []
            for i in range(0, total_samples, infer_batch):
                batch = selected_crops[i:i + infer_batch]
                output = self.segmentation_model(batch)
                seg.append(output)
            # Concatenate all results
            seg = torch.cat(seg, dim=0)

            zeros = torch.zeros([b, self.args.out_channels, x, y, z]).to(input_images.device)
            seg_out = aggregate_segmentation_with_coords(zeros, seg, selected_coord)

        else:
            seg_out = torch.zeros([b, self.args.out_channels, x, y, z]).to(input_images.device)

        # seg_out: segmentation for the whole volume
        # labels: for glance (num_crops, 2)
        # actions: (num_crops, 2)
        return seg_out, labels, actions

    def view_features(self, input_images, input_labels):
        b, _, x, y, z = input_images.size()
        assert b == 1, print('validation, batch_size should be 1, current batch_size is: ', b)

        # actions.shape: (crops, 2)
        crops, coords_all, labels, actions = self.glance_model.forward_with_labels(input_images, input_labels, training=False)

        # inference with batch, to avoid large batch for segmentation --> out of memory
        infer_batch = 4
        total_samples = crops.size()[0]

        seg = []
        features = []
        for i in range(0, total_samples, infer_batch):
            batch = crops[i:i + infer_batch]
            output,feature = self.segmentation_model(batch)
            seg.append(output)
            features.append(feature)
        # Concatenate all results
        seg = torch.cat(seg, dim=0)
        features = torch.cat(features, dim=0)

        return crops, seg, labels, actions, features

    def view_optimal(self, input_images, input_labels):
        b, _, x, y, z = input_images.size()
        assert b == 1, print('validation, batch_size should be 1, current batch_size is: ', b)

        # actions.shape: (crops, 2)
        crops, coords_all, labels, actions = self.glance_model.forward_with_labels(input_images, input_labels, training=False)

        # select by actions
        actions_argmax = actions.argmax(1)

        if actions_argmax.sum() > 0:
            selected_index = (actions_argmax == 1)
            # print('selected_index:', selected_index)

            selected_crops = crops[selected_index]
            # print('crops.shape, selected_crops.shape: ', crops.shape, selected_crops.shape)

            # Convert coord to a tensor first
            coord = torch.tensor(coords_all[0]).to(input_images.device)
            selected_coord = coord[selected_index]

            # inference with batch, to avoid large batch for segmentation --> out of memory
            infer_batch = 4
            total_samples = selected_crops.size()[0]
            seg = []
            for i in range(0, total_samples, infer_batch):
                batch = selected_crops[i:i + infer_batch]
                output = self.segmentation_model(batch)
                seg.append(output)
            # Concatenate all results
            seg = torch.cat(seg, dim=0)

            zeros = torch.zeros([b, self.args.out_channels, x, y, z]).to(input_images.device)
            seg_out = aggregate_segmentation_with_coords(zeros, seg, selected_coord)

        else:
            seg_out = torch.zeros([b, self.args.out_channels, x, y, z]).to(input_images.device)

        # seg_out: segmentation for the whole volume
        # labels: for glance (num_crops, 2)
        # actions: (num_crops, 2)
        return selected_crops, seg, seg_out, labels, actions

    def valid(self, input_images):
        b, _, x, y, z = input_images.size()
        assert b == 1, print('validation, batch_size should be 1, current batch_size is: ', b)

        crops, coords_all, actions = self.glance_model(input_images)  # actions.shape: (crops, 2)

        # select by actions
        actions_argmax = (actions.softmax(1)[:, 1] > 0.9).long()

        if actions_argmax.sum() > 0:
            selected_index = (actions_argmax == 1)

            selected_crops = crops[selected_index]
            compress_ratio = selected_crops.shape[0]/crops.shape[0]

            # Convert coord to a tensor first
            coord = torch.tensor(coords_all[0]).to(input_images.device)
            selected_coord = coord[selected_index]

            # inference with batch, to avoid large batch for segmentation --> out of memory
            infer_batch = 4
            total_samples = selected_crops.size()[0]
            seg = []
            for i in range(0, total_samples, infer_batch):
                batch = selected_crops[i:i + infer_batch]
                output = self.segmentation_model(batch)
                output = output.softmax(1)
                seg.append(output)
            # Concatenate all results
            seg = torch.cat(seg, dim=0)

            zeros = torch.zeros([b, self.args.out_channels, x, y, z]).to(input_images.device)
            seg_out = aggregate_segmentation_with_coords(zeros, seg, selected_coord)

        else:
            compress_ratio = 0.0
            seg_out = torch.zeros([b, self.args.out_channels, x, y, z]).to(input_images.device)

        return seg_out, compress_ratio


def get_dice_reward(pred: torch.Tensor, label: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    FOR ABLATION
    Compute Dice Reward per sample using argmax for prediction.
    for ablation studies

    Args:
        pred:   Raw prediction logits (b, 2, h, w, d), without softmax
        label:  Binary ground truth (b, 1, h, w, d), values 0 or 1
        smooth: Smoothing factor to avoid division by zero

    Returns:
        dice:   Dice score per sample (b,)
    """
    # Convert label to match pred shape (b, 1, h, w, d) -> (b, h, w, d)
    label_squeezed = label.squeeze(1)  # (b, h, w, d)

    # Get predicted class using argmax
    pred_classes = torch.argmax(pred, dim=1)  # (b, h, w, d)

    # Create binary foreground mask for predictions (1 where predicted as foreground)
    pred_fg = (pred_classes == 1).float()  # (b, h, w, d)

    # Create binary foreground mask for labels
    label_fg = label_squeezed.float()  # (b, h, w, d)

    # Compute intersection and volumes
    intersection = torch.sum(pred_fg * label_fg, dim=(1, 2, 3))  # (b,)
    pred_volume = torch.sum(pred_fg, dim=(1, 2, 3))  # (b,)
    label_volume = torch.sum(label_fg, dim=(1, 2, 3))  # (b,)

    # Compute Dice score
    dice_reward = (2. * intersection + smooth) / (pred_volume + label_volume + smooth)  # (b,)

    return dice_reward


def get_detection_reward(pred: torch.Tensor, label: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Compute detection Reward per sample using argmax for prediction.

    Args:
        pred:   Raw prediction logits (b, 2, h, w, d), without softmax
        label:  Binary ground truth (b, 1, h, w, d), values 0 or 1
        smooth: Smoothing factor to avoid division by zero

    Returns:
        dice:   Dice score per sample (b,)
    """
    # Convert label to match pred shape (b, 1, h, w, d) -> (b, h, w, d)
    label_squeezed = label.squeeze(1)  # (b, h, w, d)

    # Get predicted class using argmax
    pred_classes = torch.argmax(pred, dim=1)  # (b, h, w, d)

    # Create binary foreground mask for predictions (1 where predicted as foreground)
    pred_fg = (pred_classes == 1).float()  # (b, h, w, d)

    # Create binary foreground mask for labels
    label_fg = label_squeezed.float()  # (b, h, w, d)

    ### Compute intersection and volumes
    intersection = torch.sum(pred_fg * label_fg, dim=(1, 2, 3))  # (b,)
    reward = (intersection > 0).float()

    return reward


def group_relative_advantages(rewards):
    std = rewards.std()
    std = std if std > 1e-2 else 1e-2
    advantages = (rewards - rewards.mean()) / std
    advantages = torch.clamp(advantages, -5, 5)  # 限制在[-5, 5]之间
    return advantages


def get_glance_label(label):
    """
    Optimized processing of label tensor [n, 1, h, w, z]
    Returns tensor where:
        - 0: background (all zeros)
        - 1: contains cancer (any 1s)

    Args:
        label: Input tensor of shape [n, 1, h, w, z]
    Returns:
        Processed tensor of shape [n, h*w*z] with values 0, 1
    """
    # Reshape to [n, h*w*z]
    b, _, x, y, z = label.size()
    label_flat = label.view(b, x * y * z)

    # Create masks for each condition
    has_cancer = (label_flat == 1).any(dim=1, keepdim=True)

    # Combine conditions
    result = torch.zeros_like(label_flat[:, :1])  # Initialize with background
    result[has_cancer] = 1

    return result.long()


def get_glance_loss(actions, label):
    # input actions: (num_crops, 2)
    actions = actions.softmax(1)[:, 1]

    # label: 0 background, 1 cancer
    pos_label = (label == 1).long()
    neg_label = (label != 1).long()

    pos_loss = - (pos_label * torch.log(actions + 1e-6)).sum() / (pos_label.sum() + 1e-6)
    neg_loss = - (neg_label * torch.log(1 - actions + 1e-6)).sum() / (neg_label.sum() + 1e-6)

    loss = pos_loss + neg_loss

    return loss


def aggregate_segmentation_with_coords(zeros, seg, coord):
    """

    Args:
        zeros: [b, c, x, y, z]
        seg: [num_patches, c, patch_x, patch_y, patch_z]
        coord: [num_patches, 3]
    """
    b, c, x, y, z = zeros.size()
    device = zeros.device
    dtype = zeros.dtype

    count_map = torch.zeros([b, c, x, y, z], dtype=dtype, device=device)

    for i in range(seg.shape[0]):
        patch_size = seg[i].shape[1:]  # [patch_x, patch_y, patch_z]

        if len(patch_size) != 3:
            print(f"Warning: patch_size has {len(patch_size)} dimensions, expected 3. patch_size: {patch_size}")
            continue

        if coord[i].shape[0] == 3:
            x_start, y_start, z_start = coord[i]
            x_end = x_start + patch_size[0]
            y_end = y_start + patch_size[1]
            z_end = z_start + patch_size[2]
        elif coord[i].shape[0] == 6:
            x_start, y_start, z_start, x_end, y_end, z_end = coord[i]
        else:
            print(f"Warning: coord[i] has {coord[i].shape[0]} dimensions, expected 3 or 6. coord[i]: {coord[i]}")
            continue

        x_start = max(0, int(x_start))
        y_start = max(0, int(y_start))
        z_start = max(0, int(z_start))
        x_end = min(x, int(x_end))
        y_end = min(y, int(y_end))
        z_end = min(z, int(z_end))

        if x_start >= x_end or y_start >= y_end or z_start >= z_end:
            print(
                f"Warning: Invalid coordinates for patch {i}: x({x_start}:{x_end}), y({y_start}:{y_end}), z({z_start}:{z_end})")
            continue

        actual_patch_size = [x_end - x_start, y_end - y_start, z_end - z_start]
        if actual_patch_size != list(patch_size):
            seg_patch = F.interpolate(seg[i:i + 1], size=actual_patch_size, mode='trilinear', align_corners=False)
        else:
            seg_patch = seg[i:i + 1]

        weight_map = torch.ones([1, 1] + actual_patch_size, dtype=dtype, device=device)

        zeros[:, :, x_start:x_end, y_start:y_end, z_start:z_end] += seg_patch
        count_map[:, :, x_start:x_end, y_start:y_end, z_start:z_end] += weight_map

    count_map[count_map == 0] = 1
    final_result = zeros / count_map
    return final_result

