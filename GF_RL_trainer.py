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

import os
import shutil
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather
import torch.nn.functional as F
from monai.data import decollate_batch


def train_epoch(model, loader, optimizers, scheduler, scaler, epoch, loss_func, args, old_policy_model):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()

    optimizer_seg, optimizer_glance = optimizers

    old_policy_model.eval()

    for idx, batch_data in enumerate(loader):

        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        data, target = data.cuda(args.rank), target.cuda(args.rank)

        optimizer_seg.zero_grad()
        optimizer_glance.zero_grad()

        seg, seg_labels, actions, labels, gl_loss, RL_loss = model(data, target, old_policy_model)

        # segmentation loss
        seg_loss = loss_func(seg, seg_labels)
        # total loss
        total_loss = seg_loss + gl_loss * 0.1 + RL_loss

        if args.amp:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer_seg)
            scaler.step(optimizer_glance)
            scaler.update()
        else:
            total_loss.backward()
            optimizer_seg.step()
            optimizer_glance.step()

        run_loss.update(total_loss.item(), n=args.batch_size)

        lr = optimizer_seg.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        length = len(loader) // 4
        if args.rank == 0 and (idx + 1) % length == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "lr: {:.8f}".format(lr),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg, (optimizer_seg, optimizer_glance)


def val_epoch(model, loader, epoch, acc_func, args, post_label=None, post_pred=None):
    model.eval()
    run_acc = AverageMeter()
    start_time = time.time()
    with torch.no_grad():

        metric_count = 0
        for idx, batch_data in enumerate(loader):
            torch.cuda.empty_cache()

            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]

            data, target = data.cuda(args.rank), target.cuda(args.rank)

            with autocast(enabled=args.amp):
                seg, output_labels, actions = model.valid_with_labels(data, target)

            # For Glance
            # For Segmentation, target for segmentation evaluation
            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(seg)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_convert)
            seg_acc, not_nans = acc_func.aggregate()
            seg_acc = seg_acc.cuda(args.rank)
            if args.distributed:
                acc_list, not_nans_list = distributed_all_gather(
                    [seg_acc, not_nans], out_numpy=True, is_valid=idx < loader.sampler.valid_length
                )
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)
            else:
                run_acc.update(seg_acc.cpu().numpy(), n=not_nans.cpu().numpy())

            if args.rank == 0:
                seg_dice = np.mean(run_acc.avg)
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    "seg_dice: {:.4f}".format(seg_dice),
                    "time {:.2f}s".format(time.time() - start_time),
            )
            start_time = time.time()

    return run_acc.avg


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
        model,
        train_loader,
        val_loader,
        optimizers,
        loss_func,
        acc_func,
        args,
        scheduler=None,
        start_epoch=0,
        post_label=None,
        post_pred=None,
):
    from models.glance import Glance_model
    old_policy_model = Glance_model(args.glance_backbone)
    old_policy_model.cuda(args.gpu)
    old_policy_model.load_state_dict(model.glance_model.state_dict(), strict=True)
    old_policy_model.eval()

    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0
    optimizer_seg, optimizer_glance = optimizers

    for epoch in range(start_epoch, args.max_epochs):
        if epoch % args.update_epochs == 0:
            old_policy_model.load_state_dict(model.glance_model.state_dict(), strict=True)
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()

        train_loss, (optimizer_seg, optimizer_glance) = train_epoch(
            model, train_loader, (optimizer_seg, optimizer_glance), scheduler, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args, old_policy_model=old_policy_model
        )
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        b_new_best = False

        if args.rank == 0 and args.logdir is not None:
            save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")

        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_avg_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
            )

            val_avg_acc = np.mean(val_avg_acc)

            if args.rank == 0:
                print(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "dice",
                    val_avg_acc,
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max,
                        )
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))

        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max