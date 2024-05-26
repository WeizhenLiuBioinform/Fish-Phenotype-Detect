import json
import os
import datetime

import torch
from torch.utils import data
import numpy as np

import transforms
from model import HighResolutionNet
from my_dataset_coco import CocoKeypoint
from train_utils import train_eval_utils as utils
from torch.utils.tensorboard import SummaryWriter
from train_utils import WarmupScheduler
import copy

def update_kps_weights(coco_info, num_joints, old_weights, momentum=0.9):
    for result in coco_info:
        if result["pmp_threshold"] == 0.1:
            # Apply softmax to the inverted PMP
            inverse_pmp = 1 / (np.array([result["keypoints_pmp"][kp] for kp in result["keypoints_pmp"]]) + 0.01)
            exp_scores = np.exp(inverse_pmp)
            new_weights = exp_scores / np.sum(exp_scores)
            new_weights = new_weights

            # Update weights using momentum
            updated_weights = momentum * old_weights + (1 - momentum) * new_weights
            return updated_weights.reshape((num_joints,))
    return old_weights


def create_model(num_joints, load_pretrain_weights=True):
    model = HighResolutionNet(base_channel=48, num_joints=num_joints)
    print(model)

    if load_pretrain_weights:
        weights_dict = torch.load("./hrnet_w48.pth", map_location='cpu')

        for k in list(weights_dict.keys()):
            if ("head" in k) or ("fc" in k):
                del weights_dict[k]

            if "final_layer" in k:
                if weights_dict[k].shape[0] != num_joints:
                    del weights_dict[k]

        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0:
            print("missing_keys: ", missing_keys)

    return model


def main(args):
    writer = SummaryWriter('runs/hbjl_200_new_new/hbjl_3')  # You can change the experiment name
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    old_weights = np.ones(args.num_joints, dtype=np.float32) * 0.0454545454545455
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    with open(args.keypoints_path, "r") as f:
        person_kps_info = json.load(f)

    fixed_size = args.fixed_size
    heatmap_hw = (args.fixed_size[0] // 4, args.fixed_size[1] // 4)
    kps_weights = np.array(person_kps_info["kps_weights"], dtype=np.float32).reshape((args.num_joints,))
    data_transform = {
        "train": transforms.Compose([
            transforms.AffineTransform(scale=(0.65, 1.35), rotation=(-45, 45), fixed_size=fixed_size),
            transforms.KeypointToHeatMap(heatmap_hw=heatmap_hw, gaussian_sigma=2, keypoints_weights=kps_weights),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.AffineTransform(scale=None, fixed_size=fixed_size),
            transforms.KeypointToHeatMap(heatmap_hw=heatmap_hw, gaussian_sigma=2, keypoints_weights=kps_weights),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    data_root = args.data_path

    train_dataset = CocoKeypoint(data_root, "train", transforms=data_transform["train"], fixed_size=args.fixed_size)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    train_data_loader = data.DataLoader(train_dataset,
                                        batch_size=1,
                                        shuffle=True,
                                        pin_memory=True,
                                        num_workers=0,
                                        collate_fn=train_dataset.collate_fn)

    val_dataset = CocoKeypoint(data_root, "val", transforms=data_transform["val"], fixed_size=args.fixed_size,
                               det_json_path=args.person_det)
    val_data_loader = data.DataLoader(val_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=0,
                                      collate_fn=val_dataset.collate_fn)

    model = create_model(num_joints=args.num_joints)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.5, 0.5), weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    warmup_scheduler = WarmupScheduler(optimizer, warmup_epochs=args.warmup_epochs, initial_lr=args.warmup_initial_lr)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.99, patience=5, verbose=True)

    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        print("the training process from epoch{}...".format(args.start_epoch))

    train_loss = []
    learning_rate = []
    val_map = []
    best_pmp_avg = float('-inf')
    best_model_weights = None

    for epoch in range(args.start_epoch, args.epochs):
        data_transform["train"] = transforms.Compose([
            transforms.AffineTransform(scale=(0.65, 1.35), rotation=(-45, 45), fixed_size=fixed_size),
            transforms.KeypointToHeatMap(heatmap_hw=heatmap_hw, gaussian_sigma=2, keypoints_weights=kps_weights),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_dataset = CocoKeypoint(data_root, "train", transforms=data_transform["train"], fixed_size=args.fixed_size)
        train_data_loader = data.DataLoader(train_dataset,
                                            batch_size=1,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=0,
                                            collate_fn=train_dataset.collate_fn)

        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device=device, epoch=epoch,
                                              print_freq=50, warmup=True,
                                              scaler=scaler)
        writer.add_scalar('Training Loss', mean_loss.item(), epoch)
        writer.add_scalar('Learning Rate', lr, epoch)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        coco_info = utils.evaluate(model, val_data_loader, device=device,
                                   flip=False, flip_pairs=None)

        if epoch > 1000:
            new_weights = update_kps_weights(coco_info, args.num_joints, old_weights)
        else:
            new_weights = None
        if new_weights is not None:
            kps_weights = new_weights
            old_weights = new_weights

        current_pmp_avg = None
        for result in coco_info:
            if result["pmp_threshold"] == 0.1:
                current_pmp_avg = result["average_pmp"]
                break
        if current_pmp_avg is not None and current_pmp_avg > best_pmp_avg:
            best_pmp_avg = current_pmp_avg
            best_model_weights = copy.deepcopy(model.state_dict())
            save_files = {
                'model': best_model_weights,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch}
            if args.amp:
                save_files["scaler"] = scaler.state_dict()
            torch.save(save_files, os.path.join(args.output_dir, f"best_model.pth"))

        current_pmp = None
        for result in coco_info:
            if result["pmp_threshold"] == 0.1:
                current_pmp = result["average_pmp"]
                break

        for result in coco_info:
            pmp_threshold = result["pmp_threshold"]
            average_pmp = result["average_pmp"]
            writer.add_scalar(f'PMP/Average/Threshold_{pmp_threshold}', average_pmp, epoch)
            for keypoint_name, pmp_value in result["keypoints_pmp"].items():
                writer.add_scalar(f'PMP/{keypoint_name}/Threshold_{pmp_threshold}', pmp_value, epoch)

        with open(results_file, 'w') as file:
            for result in coco_info:
                pmp_threshold = result["pmp_threshold"]
                average_pmp = result["average_pmp"]
                file.write(f'---------------------------\n')
                file.write(f"PMP Threshold t = {pmp_threshold}, Average: {average_pmp}\n")
                file.write(f'---------------------------\n')
                for name, pmp in result["keypoints_pmp"].items():
                    file.write(f"{name}: {pmp}\n")

        if epoch % 10 == 0:
            save_files = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch}
            if args.amp:
                save_files["scaler"] = scaler.state_dict()

            torch.save(save_files, os.path.join(args.output_dir, f"model-{epoch}.pth"))

        original_loss, additional_loss, total_loss = utils.val_one_epoch(model, val_data_loader, device, scaler=scaler)

        writer.add_scalar('Validation Original Loss', original_loss, epoch)
        writer.add_scalar('Validation Additional Loss', additional_loss, epoch)
        writer.add_scalar('Validation Total Loss', total_loss, epoch)

        if epoch < args.warmup_epochs:
            warmup_scheduler.step()
        else:
            lr_scheduler.step(current_pmp)

    writer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--device', default='cuda:1', help='device')
    parser.add_argument('--data-path', default='/home/tanjy/data/new_data_200/hbjl_new_new', help='dataset')
    parser.add_argument('--keypoints-path', default="./person_keypoints.json", type=str,
                        help='person_keypoints.json path')
    parser.add_argument('--person-det', type=str, default=None)
    parser.add_argument('--fixed-size', default=[1728, 2304], nargs='+', type=int, help='input size')
    parser.add_argument('--num-joints', default=22, type=int, help='num_joints')
    parser.add_argument('--output-dir', default='./save_weights/hbjl_200_new_new/hbjl_3', help='path where to save')
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--batch-size', default=2, type=int, metavar='N',
                        help='batch size when training.')
    parser.add_argument('--warmup-epochs', default=20, type=int, help='number of warmup epochs')
    parser.add_argument('--warmup-initial-lr', default=1e-5, type=float, help='initial learning rate for warmup')
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
