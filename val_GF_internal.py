
# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from torch.cuda.amp import autocast
# from monai.data import decollate_batch
from monai.metrics import DiceMetric
from monai.transforms import *
from monai.utils.enums import MetricReduction
from utils.utils import *
from utils.utils import AverageMeter
from monai.data import *
import resource
from models.GF_Screen_RL import GFScreen_RL
import time

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))
print('Setting resource limit:', str(resource.getrlimit(resource.RLIMIT_NOFILE)))

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '28890'

parser = argparse.ArgumentParser(description="Segmentation pipeline")
parser.add_argument(
    "--test_data_path", default="./data/imagesTr/", type=str, help="test_data_path")
parser.add_argument(
    "--test_label_path", default="./data/labelsTr/", type=str, help="test_data_path")
parser.add_argument(
    "--save_prediction_path", default="./pred/GF/", type=str, help="test_prediction_path")
parser.add_argument(
    "--trained_pth", default="./runs/logs/model_final.pt", type=str, help="trained checkpoint directory")
parser.add_argument("--glance_backbone", default='resnet18', help="pre-trained root")

parser.add_argument("--data_dir", default="./data", type=str, help="dataset directory")
parser.add_argument("--cache_dir", default='./data/cache', type=str, help="dataset json file")
parser.add_argument("--use_persistent_dataset", default=True, help="use monai Dataset class")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=2, type=int, help="number of output channels")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.0, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.0, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=3.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=64, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", default=True, help="use gradient checkpointing to save memory")


def main():
    args = parser.parse_args()
    from utils.data_utils import get_val_loader
    test_loader, test_transforms = get_val_loader(args)

    log_path = os.path.join(args.save_prediction_path, "eval_log.txt")
    os.makedirs(args.save_prediction_path, exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8")
    log_file.write(f"{args}\n")

    model = GFScreen_RL(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dict = torch.load(args.trained_pth)["state_dict"]
    model.load_state_dict(model_dict, strict=True)

    model.eval()
    model.to(device)

    # enable cuDNN benchmark
    torch.backends.cudnn.benchmark = True

    post_transforms = Compose([EnsureTyped(keys=["pred"]),
                               Invertd(keys=["pred"],
                                       transform=test_transforms,
                                       orig_keys="image",
                                       meta_keys="pred_meta_dict",
                                       orig_meta_keys="image_meta_dict",
                                       meta_key_postfix="meta_dict",
                                       nearest_interp=True,
                                       to_tensor=True),
                               AsDiscreted(keys="pred", argmax=False, to_onehot=None),
                               SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=args.save_prediction_path,
                                          separate_folder=False, folder_layout=None,
                                          resample=False),
                               ])

    acc_func = DiceMetric(include_background=False, reduction=MetricReduction.MEAN, get_not_nans=True)
    run_acc = AverageMeter()
    post_label = AsDiscrete(to_onehot=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)

    total_compress_ratio = 0.0
    count = 0

    overall_duration = 0

    with torch.no_grad():
        for idx, batch_data in enumerate(test_loader):
            torch.cuda.empty_cache()
            start_time = time.time()

            data = batch_data["image"]
            data = data.cuda()

            label = batch_data["label"]
            label = label.cuda()

            name = batch_data['name'][0]
            print(name)
            log_file.write(f"{name}\n")

            with autocast(enabled=True):
                logits, compress_ratio = model.valid(data)

            total_compress_ratio += compress_ratio
            count += 1
            mean_compress_ratio = total_compress_ratio / count
            print('compress_ratio:', mean_compress_ratio)
            log_file.write(f"compress_ratio: {compress_ratio}\n")

            duration = time.time() - start_time
            overall_duration += duration
            average_duration = overall_duration / count

            val_labels_list = decollate_batch(label)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            case_dice = dice(logits.argmax(1).data.cpu().numpy(), label.data.cpu().numpy())
            print('Case dice:', case_dice)
            log_file.write(f"Case dice: {case_dice}\n")

            print('Average Dice:', np.mean(run_acc.avg))
            log_file.write(f"Average Dice: {np.mean(run_acc.avg)}\n")

            print('Count:', count)
            log_file.write(f"Count: {count}\n")

            print('Duration:', duration)
            log_file.write(f"Duration: {duration}\n")

            print('Average Duration:', average_duration)
            log_file.write(f"Average Duration: {average_duration}\n\n")

            print('\n')

            output = logits.argmax(1)
            batch_data['pred'] = output.unsqueeze(1)

            batch_data = [post_transforms(i) for i in
                          decollate_batch(batch_data)]

            os.rename(os.path.join(args.save_prediction_path, name + '_0000_trans.nii.gz'), os.path.join(args.save_prediction_path, name + '.nii.gz'))


if __name__ == "__main__":
    main()
    args = parser.parse_args()

    check_each_dataset_lesion_dice(pred_path=args.save_prediction_path, label_path=args.test_label_path)