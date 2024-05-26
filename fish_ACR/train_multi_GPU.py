import json
import time
import os
import datetime

import torch
from torch.utils import data
import numpy as np

import transforms
from model import HighResolutionNet
from my_dataset_coco import CocoKeypoint
import train_utils.train_eval_utils as utils
from train_utils import init_distributed_mode, save_on_master, mkdir


def create_model(num_joints, load_pretrain_weights=True):
    model = HighResolutionNet(base_channel=32, num_joints=num_joints)

    if load_pretrain_weights:
        # Load pretrained model weights
        # Link: https://pan.baidu.com/s/1Lu6mMAWfm_8GGykttFMpVw Extraction code: f43o
        weights_dict = torch.load("./hrnet_w32.pth", map_location='cpu')

        for k in list(weights_dict.keys()):
            # Delete unnecessary weights if loading ImageNet weights
            if ("head" in k) or ("fc" in k):
                del weights_dict[k]

            # If loading COCO weights, compare num_joints and delete if not equal
            if "final_layer" in k:
                if weights_dict[k].shape[0] != num_joints:
                    del weights_dict[k]

        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0:
            print("missing_keys: ", missing_keys)

    return model


def main(args):
    init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # File to save coco_info
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    key_results_file = f"results{now}.txt"

    with open(args.keypoints_path, "r") as f:
        person_kps_info = json.load(f)

    fixed_size = args.fixed_size
    heatmap_hw = (args.fixed_size[0] // 4, args.fixed_size[1] // 4)
    kps_weights = np.array(person_kps_info["kps_weights"],
                           dtype=np.float32).reshape((args.num_joints,))
    data_transform = {
        "train": transforms.Compose([
            transforms.HalfBody(0.3, person_kps_info["upper_body_ids"], person_kps_info["lower_body_ids"]),
            transforms.AffineTransform(scale=(0.65, 1.35), rotation=(-45, 45), fixed_size=fixed_size),
            transforms.RandomHorizontalFlip(0.5, person_kps_info["flip_pairs"]),
            transforms.KeypointToHeatMap(heatmap_hw=heatmap_hw, gaussian_sigma=2, keypoints_weights=kps_weights),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=fixed_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    data_root = args.data_path

    # load train data set
    # coco2017 -> annotations -> person_keypoints_train2017.json
    train_dataset = CocoKeypoint(data_root, "train", transforms=data_transform["train"], fixed_size=args.fixed_size)

    # load validation data set
    # coco2017 -> annotations -> person_keypoints_val2017.json
    val_dataset = CocoKeypoint(data_root, "val", transforms=data_transform["val"], fixed_size=args.fixed_size,
                               det_json_path=None)

    print("Creating data loaders")
    if args.distributed:
        train_sampler = data.distributed.DistributedSampler(train_dataset)
        test_sampler = data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = data.RandomSampler(train_dataset)
        test_sampler = data.SequentialSampler(val_dataset)

    train_batch_sampler = data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

    data_loader = data.DataLoader(train_dataset,
                                  batch_sampler=train_batch_sampler,
                                  num_workers=args.workers,
                                  collate_fn=train_dataset.collate_fn)

    data_loader_test = data.DataLoader(val_dataset,
                                       batch_size=args.batch_size,
                                       sampler=test_sampler,
                                       num_workers=args.workers,
                                       collate_fn=train_dataset.collate_fn)

    print("Creating model")
    # create model num_classes equal background + classes
    model = create_model(num_joints=args.num_joints)
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    # If resume parameter is passed, continue training from previous checkpoint
    if args.resume:
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.
        checkpoint = torch.load(args.resume, map_location='cpu')  # Load previously saved weight file (including optimizer and learning rate strategy)
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        utils.evaluate(model, data_loader_test, device=device,
                       flip=True, flip_pairs=person_kps_info["flip_pairs"])
        return

    train_loss = []
    learning_rate = []
    val_map = []

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        mean_loss, lr = utils.train_one_epoch(model, optimizer, data_loader,
                                              device, epoch, args.print_freq,
                                              warmup=True, scaler=scaler)

        # update learning rate
        lr_scheduler.step()

        # evaluate after every epoch
        key_info = utils.evaluate(model, data_loader_test, device=device,
                                  flip=True, flip_pairs=person_kps_info["flip_pairs"])

        # Only perform write operations on the main process
        if args.rank in [-1, 0]:
            train_loss.append(mean_loss.item())
            learning_rate.append(lr)
            val_map.append(key_info[1])  # @0.5 mAP

            # write into txt
            with open(key_results_file, "a") as f:
                # Write data including coco indicators, loss and learning rate
                result_info = [f"{i:.4f}" for i in key_info + [mean_loss.item()]] + [f"{lr:.6f}"]
                txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                f.write(txt + "\n")

        if args.output_dir:
            # Only perform save weight operations on the main process
            save_files = {'model': model_without_ddp.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'lr_scheduler': lr_scheduler.state_dict(),
                          'args': args,
                          'epoch': epoch}
            if args.amp:
                save_files["scaler"] = scaler.state_dict()
            save_on_master(save_files,
                           os.path.join(args.output_dir, f'model_{epoch}.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if args.rank in [-1, 0]:
        # plot loss and lr curve
        if len(train_loss) != 0 and len(learning_rate) != 0:
            from plot_curve import plot_loss_and_lr
            plot_loss_and_lr(train_loss, learning_rate)

        # plot mAP curve
        if len(val_map) != 0:
            from plot_curve import plot_map
            plot_map(val_map)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # Root directory of training files (coco2017)
    parser.add_argument('--data-path', default='/home/langy/pangtouyu', help='dataset')
    # Training device type
    parser.add_argument('--device', default='cuda', help='device')
    # COCO dataset human keypoint information
    parser.add_argument('--keypoints-path', default="./person_keypoints.json", type=str,
                        help='person_keypoints.json path')
    # Validation set person detection information provided by the original project.
    # If you want to use GT information, simply set this parameter to None. It is recommended to set it to None.
    parser.add_argument('--person-det', type=str, default=None)
    parser.add_argument('--fixed-size', default=[1152, 864], nargs='+', type=int, help='input size')
    # Number of detection target categories (excluding background)
    parser.add_argument('--num-joints', default=22, type=int, help='num_joints(num_keypoints)')
    # Batch size per GPU
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    # Specify which epoch number to start training from
    parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')
    # Total number of epochs to run
    parser.add_argument('--epochs', default=210, type=int, metavar='N',
                        help='number of total epochs to run')
    # Number of threads for data loading and preprocessing
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # Learning rate
    parser.add_argument('--lr', default=0.001, type=float,
                        help='initial learning rate, 0.001 is the default value for training '
                             'on 4 gpus and 32 images_per_gpu')
    # AdamW's weight_decay parameter
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # Parameters for torch.optim.lr_scheduler.MultiStepLR
    parser.add_argument('--lr-steps', default=[170, 200], nargs='+', type=int,
                        help='decrease lr every step-size epochs')
    # Parameters for torch.optim.lr_scheduler.MultiStepLR
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    # Frequency of printing training process information
    parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
    # File save address
    parser.add_argument('--output-dir', default='./multi_train', help='path where to save')
    # Continue training based on the previous training results
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--test-only', action="store_true", help="test only")

    # Number of processes to be started (note that it is not threads)
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--sync-bn", action="store_true", help="Use sync batch norm")
    # Whether to use mixed precision training (requires GPU support for mixed precision)
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    # If the save file address is specified, check if the folder exists, if not, create it
    if args.output_dir:
        mkdir(args.output_dir)

    main(args)
