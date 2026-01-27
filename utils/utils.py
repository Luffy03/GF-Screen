import numpy as np
import scipy.ndimage as ndimage
import torch
import os
import SimpleITK as sitk
from tqdm import tqdm
from scipy.ndimage import label as connect_label


def read(img, transpose=False):
    img = sitk.ReadImage(img)
    direction = img.GetDirection()
    origin = img.GetOrigin()
    Spacing = img.GetSpacing()

    img = sitk.GetArrayFromImage(img)
    if transpose:
        img = img.transpose(1, 2, 0)

    return img, direction, origin, Spacing


def check_acc_volume_level(pred_path, label_path):
    ls = os.listdir(pred_path)
    total_num = 0
    TP, TN, FP, FN = 0, 0, 0, 0

    for i in tqdm(ls):
        if i.endswith('.nii.gz'):
            # lesion type wise
            if 'MSD_colon' in i:
                pred = read(os.path.join(pred_path, i))[0]
                label = read(os.path.join(label_path, i))[0]
                total_num += 1

                if label.sum() > 0:
                    if (pred*label).sum() > 0:
                        TP += 1
                    else:
                        FN += 1
                else:
                    if pred.sum() > 0:
                        FP += 1
                    else:
                        TN += 1
    print('TP, TN, FP, FN:', TP, TN, FP, FN)
    accuracy = (TP+TN)/total_num
    sensitivity = TP / (TP + FN + 1e-6)
    specificity = TN / (TN + FP + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    print('accuracy, sensitivity, specificity, precision, recall, f1_score: ',
          accuracy, sensitivity, specificity, precision, recall, f1_score)


def check_acc_case_level(pred_path, label_path):
    ls = os.listdir(pred_path)
    total_num = 0
    TP, TN, FP, FN = 0, 0, 0, 0

    for i in tqdm(ls):
        if i.endswith('.nii.gz'):
            pred = read(os.path.join(pred_path, i))[0]
            label = read(os.path.join(label_path, i))[0]

            if label.sum() > 0:
                labeled_matrix, num_features = connect_label(label)
                for i in range(1, num_features + 1):
                    total_num += 1

                    cur_case = labeled_matrix.copy()
                    cur_case[labeled_matrix == i] = 1
                    cur_case[labeled_matrix != i] = 0
                    # print('cur_case: ', cur_case.shape, np.unique(cur_case))

                    if (pred*cur_case).sum() > 0:
                        TP += 1
                    else:
                        FN += 1

            else:
                total_num += 1
                if pred.sum() > 125:
                    FP += 1
                else:
                    TN += 1

    print('TP, TN, FP, FN:', TP, TN, FP, FN)
    accuracy = (TP+TN)/total_num
    sensitivity = TP / (TP + FN + 1e-6)
    specificity = TN / (TN + FP + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    print('accuracy, sensitivity, specificity, precision, recall, f1_score: ',
          accuracy, sensitivity, specificity, precision, recall, f1_score)


def check_each_dataset_lesion_dice(pred_path, label_path):
    ls = os.listdir(pred_path)
    dataset_keys = {
                'Adrenal': [],
                'Chest_coronacases': [],
                'Chest_LIDC-IDRI': [],
                'Chest_MSD_lung': [],
                'Chest_NSCLC-Radiogenomics': [],
                'Chest_NSCLC-Radiomics': [],
                'Chest_volume-covid19': [],
                'Chest_NSCLCPleuralEffusion': [],
                'HCC': [],
                'KiTS23': [],
                'MSD_colon': [],
                'MSD_hepaticvessel': [],
                'MSD_liver': [],
                'MSD_pancreas': [],
                'Panorama': [],
                'WAWTACE': []
        }

    lesion_keys = {
        'Lung tumor': [],
        'Lung nodule': [],
        'Pleural effusion': [],
        'COVID-19': [],
        'Liver tumor': [],
        'Pancreas tumor': [],
        'Kidney tumor': [],
        'Adrenal carci': [],
        'Colon tumor': [],

    }

    for i in tqdm(ls):
        if i.endswith('.nii.gz'):
            pred = read(os.path.join(pred_path, i))[0]
            label = read(os.path.join(label_path, i))[0]

            case_dice = dice(pred, label)
            print(i, case_dice)

            # dataset wise
            for key in dataset_keys:
                if key in i:
                    dataset_keys[key].append(case_dice)

            # lesion type wise
            if 'Chest_MSD_lung' in i or 'Chest_NSCLC-Radiogenomics' in i or 'Chest_NSCLC-Radiomics' in i:
                lesion_keys['Lung tumor'].append(case_dice)

            if 'Chest_LIDC-IDRI' in i:
                lesion_keys['Lung nodule'].append(case_dice)

            if 'Chest_NSCLCPleuralEffusion' in i:
                lesion_keys['Pleural effusion'].append(case_dice)

            if 'Chest_volume-covid19' in i:
                lesion_keys['COVID-19'].append(case_dice)

            if 'HCC' in i or 'MSD_liver' in i or 'MSD_hepaticvessel' in i or 'WAWTACE' in i:
                lesion_keys['Liver tumor'].append(case_dice)

            if 'MSD_pancreas' in i or 'Panorama' in i:
                lesion_keys['Pancreas tumor'].append(case_dice)

            if 'KiTS23' in i:
                lesion_keys['Kidney tumor'].append(case_dice)

            if 'Adrenal' in i:
                lesion_keys['Adrenal carci'].append(case_dice)

            if 'MSD_colon' in i:
                lesion_keys['Colon tumor'].append(case_dice)

    print('per dataset')
    for key in dataset_keys:
        dice_list = dataset_keys[key]
        print(key, np.mean(dice_list))

    print('\n')
    print('per lesion type')

    for key in lesion_keys:
        lesion_list = lesion_keys[key]
        print(key, np.mean(lesion_list))



def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 1
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def distributed_all_gather(
    tensor_list, valid_batch_size=None, out_numpy=False, world_size=None, no_barrier=False, is_valid=None
):
    if world_size is None:
        world_size = torch.distributed.get_world_size()
    if valid_batch_size is not None:
        valid_batch_size = min(valid_batch_size, world_size)
    elif is_valid is not None:
        is_valid = torch.tensor(bool(is_valid), dtype=torch.bool, device=tensor_list[0].device)
    if not no_barrier:
        torch.distributed.barrier()
    tensor_list_out = []
    with torch.no_grad():
        if is_valid is not None:
            is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
            torch.distributed.all_gather(is_valid_list, is_valid)
            is_valid = [x.item() for x in is_valid_list]
        for tensor in tensor_list:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gather_list, tensor)
            if valid_batch_size is not None:
                gather_list = gather_list[:valid_batch_size]
            elif is_valid is not None:
                gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]
            if out_numpy:
                gather_list = [t.cpu().numpy() for t in gather_list]
            tensor_list_out.append(gather_list)
    return tensor_list_out


def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)