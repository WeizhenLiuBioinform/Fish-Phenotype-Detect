import os
import copy

import torch
import numpy as np
import cv2
import torch.utils.data as data
from pycocotools.coco import COCO


class CocoKeypoint(data.Dataset):
    def __init__(self,
                 root,
                 dataset="train",
                 transforms=None,
                 det_json_path=None,
                 fixed_size=(256, 192)):
        super().__init__()
        assert dataset in ["train", "val"], 'dataset must be in ["train", "val"]'
        anno_file = f"{dataset}.json"
        assert os.path.exists(root), "file '{}' does not exist.".format(root)
        self.img_root = os.path.join(root, f"{dataset}")
        assert os.path.exists(self.img_root), "path '{}' does not exist.".format(self.img_root)
        self.anno_path = os.path.join(root, "annotations", anno_file)
        assert os.path.exists(self.anno_path), "file '{}' does not exist.".format(self.anno_path)

        self.fixed_size = fixed_size
        self.mode = dataset
        self.transforms = transforms
        self.coco = COCO(self.anno_path)
        img_ids = list(sorted(self.coco.imgs.keys()))

        if det_json_path is not None:
            det = self.coco.loadRes(det_json_path)
        else:
            det = self.coco

        self.valid_person_list = []
        obj_idx = 0
        for img_id in img_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            ann_ids = det.getAnnIds(imgIds=img_id)
            anns = det.loadAnns(ann_ids)
            for ann in anns:
                # only save person class
                if ann["category_id"] != 1:
                    print(f'warning: find not support id: {ann["category_id"]}, only support id: 1 (person)')
                    continue

                # Skip checking for COCO_val2017_detections_AP_H_56_person.json files as they only contain detection information, not keypoint information
                if det_json_path is None:
                    # skip objs without keypoints annotation
                    if "keypoints" not in ann:
                        continue
                    if max(ann["keypoints"]) == 0:
                        continue

                xmin, ymin, w, h = ann['bbox']
                # Use only valid bounding boxes
                if w > 0 and h > 0:
                    info = {
                        "box": [xmin, ymin, w, h],
                        "image_path": os.path.join(self.img_root, img_info["file_name"]),
                        "image_id": img_id,
                        "image_width": img_info['width'],
                        "image_height": img_info['height'],
                        "obj_origin_hw": [h, w],
                        "obj_index": obj_idx,
                        "score": ann["score"] if "score" in ann else 1.
                    }

                    # Skip for COCO_val2017_detections_AP_H_56_person.json files as they only contain detection information, not keypoint information
                    if det_json_path is None:
                        keypoints = np.array(ann["keypoints"]).reshape([-1, 3])
                        visible = keypoints[:, 2]
                        keypoints = keypoints[:, :2]
                        info["keypoints"] = keypoints
                        info["visible"] = visible

                    self.valid_person_list.append(info)
                    obj_idx += 1

    # def __getitem__(self, idx):
    #     target = copy.deepcopy(self.valid_person_list[idx])

    #     image = cv2.imread(target["image_path"])
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     if self.transforms is not None:
    #         image, person_info = self.transforms(image, target)

    #     return image, target
    def __getitem__(self, idx):
        target = copy.deepcopy(self.valid_person_list[idx])

        image = cv2.imread(target["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        scale, angle, src_center = None, None, None  # default values

        if self.transforms is not None:
            image, target, extra_info = self.transforms(image, target)
            scale = extra_info.get('scale', None)
            angle = extra_info.get('angle', None)
            src_center = extra_info.get('src_center', None)

        return image, target, scale, angle, src_center

    def __len__(self):
        return len(self.valid_person_list)

    @staticmethod
    # def collate_fn(batch):
    #     imgs_tuple, targets_tuple = tuple(zip(*batch))
    #     imgs_tensor = torch.stack(imgs_tuple)
    #     return imgs_tensor, targets_tuple
    def collate_fn(batch):
        # Split each element in the batch
        imgs_list, targets_tuple, scales_list, angles_list, centers_list = zip(*batch)
        imgs_tensor = torch.stack(imgs_list)  # Stack all images into one tensor
        # Return targets_tuple directly as a tuple
        return imgs_tensor, targets_tuple, list(scales_list), list(angles_list), list(centers_list)


if __name__ == '__main__':
    train = CocoKeypoint("/home/tanjy/data/data_200/huabanjinli", dataset="val")
    print(len(train))
    t = train[0]
    print(t)
