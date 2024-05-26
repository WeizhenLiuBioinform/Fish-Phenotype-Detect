"""
This script is used to call trained model weights to calculate COCO metrics for the validation/test set.
"""

import os
import json

import torch
from tqdm import tqdm
import numpy as np
import datetime
from model import HighResolutionNet
from train_utils import EvalCOCOMetric
from my_dataset_coco import CocoKeypoint
import transforms

import csv

def save_as_csv(coco_info, save_name):
    with open(save_name, 'w', newline='', encoding='utf-8') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['PMP Threshold', 'Average PMP', 'Keypoint', 'PMP Value'])

        for result in coco_info:
            pmp_threshold = result["pmp_threshold"]
            average_pmp = result["average_pmp"]  # Get average PMP value
            for name, pmp in result["keypoints_pmp"].items():
                csv_writer.writerow([pmp_threshold, average_pmp, name, pmp])

def summarize(self, catId=None):
    """
    Compute and display summary metrics for evaluation results.
    Note this function can *only* be applied on the default parameter setting
    """

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, :, catId, aind, mind]
            else:
                s = s[:, :, :, aind, mind]

        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, catId, aind, mind]
            else:
                s = s[:, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        print_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return mean_s, print_string

    stats, print_list = [0] * 10, [""] * 10
    stats[0], print_list[0] = _summarize(1, maxDets=20)
    stats[1], print_list[1] = _summarize(1, maxDets=20, iouThr=.5)
    stats[2], print_list[2] = _summarize(1, maxDets=20, iouThr=.75)
    stats[3], print_list[3] = _summarize(1, maxDets=20, areaRng='medium')
    stats[4], print_list[4] = _summarize(1, maxDets=20, areaRng='large')
    stats[5], print_list[5] = _summarize(0, maxDets=20)
    stats[6], print_list[6] = _summarize(0, maxDets=20, iouThr=.5)
    stats[7], print_list[7] = _summarize(0, maxDets=20, iouThr=.75)
    stats[8], print_list[8] = _summarize(0, maxDets=20, areaRng='medium')
    stats[9], print_list[9] = _summarize(0, maxDets=20, areaRng='large')

    print_info = "\n".join(print_list)

    if not self.eval:
        raise Exception('Please run accumulate() first')

    return stats, print_info


def save_info(coco_evaluator,
              save_name: str = "record_mAP.txt"):
    # calculate COCO info for all keypoints
    coco_stats, print_coco = summarize(coco_evaluator)

    # Save validation results to a txt file
    with open(save_name, "w") as f:
        record_lines = ["COCO results:", print_coco]
        f.write("\n".join(record_lines))


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    data_transform = {
        "val": transforms.Compose([
            # transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=fixed_size),
            transforms.AffineTransform(scale=None, fixed_size=args.resize_hw),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    # read class_indict
    label_json_path = args.label_json_path
    assert os.path.exists(label_json_path), "json file {} does not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        person_coco_info = json.load(f)

    data_root = args.data_path

    # Note that the collate_fn here is custom because the data read includes images and targets,
    # and the default method cannot be used to synthesize batches directly
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    # load validation data set
    val_dataset = CocoKeypoint(data_root, "val", transforms=data_transform["val"], det_json_path=None)
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> val.txt
    # val_dataset = VOCInstances(data_root, year="2012", txt_name="val.txt", transforms=data_transform["val"])
    val_dataset_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     pin_memory=True,
                                                     num_workers=nw,
                                                     collate_fn=val_dataset.collate_fn)

    # create model
    model = HighResolutionNet(num_joints=22)

    # Load your own trained model weights
    weights_path = args.weights_path
    assert os.path.exists(weights_path), "not found {} file.".format(weights_path)
    # print(model)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'], strict=False)
    model.to(device)


    # evaluate on the val dataset
    key_metric = EvalCOCOMetric(val_dataset.coco, "keypoints", "key_results.json")
    model.eval()
    with torch.no_grad():
        for images, targets, scale, angle, src_center in tqdm(val_dataset_loader, desc="validation..."):
            # Send images to the specified device
            images = images.to(device)

            # inference
            outputs = model(images)
            if args.flip:
                flipped_images = transforms.flip_images(images)
                flipped_outputs = model(flipped_images)
                flipped_outputs = transforms.flip_back(flipped_outputs, person_coco_info["flip_pairs"])
                # feature is not aligned, shift flipped heatmap for higher accuracy
                # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/22
                flipped_outputs[..., 1:] = flipped_outputs.clone()[..., 0:-1]
                outputs = (outputs + flipped_outputs) * 0.5

            # decode keypoint
            reverse_trans = [t["reverse_trans"] for t in targets]
            outputs = transforms.get_final_preds(outputs, reverse_trans, post_processing=True)

            key_metric.update(targets, outputs)

    key_metric.synchronize_results()
    coco_info = key_metric.evaluate()
    keypoint_record = "keypoint_record_hbjl_new_new_200_1{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    with open(keypoint_record, 'w') as file:
        for result in coco_info:
            pmp_threshold = result["pmp_threshold"]
            average_pmp = result["average_pmp"]  # Get average PMP value
            file.write(f'---------------------------\n')
            file.write(f"PMP threshold t = {pmp_threshold}, average value: {average_pmp}\n")
            file.write(f'---------------------------\n')
            for name, pmp in result["keypoints_pmp"].items():
                file.write(f"{name}: {pmp}\n")

    keypoint_record1 = "keypoint_record_hbjl_new_new_200_1{}.csv".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    save_as_csv(coco_info, keypoint_record1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # Specify device type
    parser.add_argument('--device', default='cuda:0', help='device')

    parser.add_argument('--resize-hw', type=list, default=[1728, 2304], help="resize for predict")
    # parser.add_argument('--resize-hw', type=list, default=[864, 1152], help="resize for predict")
    # Whether to enable image flip
    # parser.add_argument('--flip', type=bool, default=True, help='whether using flipped images')
    parser.add_argument('--flip', type=bool, default=None, help='whether using flipped images')

    # Root directory of the dataset
    # parser.add_argument('--data-path', default='/home/tanjy/data/new_data_200/hbjl', help='dataset root')
    # parser.add_argument('--data-path', default='/home/tanjy/data/new_data_200/sby', help='dataset root')
    # parser.add_argument('--data-path', default='/home/tanjy/data/new_data_200/pty', help='dataset root')
    # parser.add_argument('--data-path', default='/home/tanjy/data/new_data_200/liyu', help='dataset root')
    # parser.add_argument('--data-path', default='/home/tanjy/data/sby_50', help='dataset root')
    # parser.add_argument('--data-path', default='/home/tanjy/data/new_data_200/sby_new', help='dataset root')
    # parser.add_argument('--data-path', default='/home/tanjy/data/new_data_200/sby_new_new', help='dataset root')
    # parser.add_argument('--data-path', default='/home/tanjy/data/new_data_200/hbjl_new', help='dataset root')
    parser.add_argument('--data-path', default='/home/tanjy/data/new_data_200/hbjl_new_new', help='dataset root')

    # Trained weights file

    # parser.add_argument('--weights-path', default='/home/tanjy/code/HRnet_copy/save_weights/sby_200_new/sby_8/best_model.pth', type=str, help='training weights')
    # parser.add_argument('--weights-path', default='/home/tanjy/code/HRnet_copy/save_weights/hbjl_200_new/hbjl_3/best_model.pth', type=str, help='training weights')
    # parser.add_argument('--weights-path', default='/home/tanjy/code/HRnet_copy/save_weights/pty_200_new/pty_7/best_model.pth', type=str, help='training weights')
    parser.add_argument('--weights-path', default='/home/tanjy/code/HRnet/save_weights/hbjl_200_new_new/hbjl_1/best_model.pth', type=str, help='training weights')
    # batch size
    parser.add_argument('--batch-size', default=1, type=int, metavar='N',
                        help='batch size when validation.')
    # Category index and category name correspondence
    parser.add_argument('--label-json-path', type=str, default="/home/tanjy/code/HRnet_copy/person_keypoints.json")
    # Validation set person detection information provided by the original project,
    # if you want to use GT information, set this parameter to None
    parser.add_argument('--person-det', type=str, default=None)

    args = parser.parse_args()

    main(args)
