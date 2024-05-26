import json
import copy

from PIL import Image, ImageDraw
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from .distributed_utils import all_gather, is_main_process
from transforms import affine_points


def merge(img_ids, eval_results):
    """Merge data from multiple processes together"""
    all_img_ids = all_gather(img_ids)
    all_eval_results = all_gather(eval_results)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_results = []
    for p in all_eval_results:
        merged_eval_results.extend(p)

    merged_img_ids = np.array(merged_img_ids)

    # keep only unique (and in sorted order) images
    # Remove duplicate image indices. During multi-GPU training, the same image may be assigned to multiple processes to ensure each process has the same number of training images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_results = [merged_eval_results[i] for i in idx]

    return list(merged_img_ids), merged_eval_results


def calculate_pmp(pred_keypoints, true_keypoints, phenotype_rules, num_keypoints, t):
    """
    Calculate the pmp value for each keypoint.
    """
    # Initialize a list to store the pmp values for each keypoint
    pmp_values = np.zeros(num_keypoints)

    """ 
    # Iterate through each dataset instance
    for d in dataset_dicts:
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # Assume predictor is defined
        pred_keypoints = outputs["instances"].pred_keypoints[0].cpu().numpy()
        true_keypoints = np.array(d["annotations"][0]["keypoints"]).reshape((num_keypoints, 3))

        # Calculate the threshold for each keypoint
        thresholds = calculate_thresholds(true_keypoints, phenotype_rules)

        # Calculate the pmp for each keypoint
        for i in range(num_keypoints):
            pmp_values[i] += np.linalg.norm(pred_keypoints[i, :2] - true_keypoints[i, :2]) <= thresholds[i] * t 
    """
    # Assume pred_keypoints and true_keypoints have shape (40, 66)
    for pred_kp, true_kp in zip(pred_keypoints, true_keypoints):
        # pred_kp now contains a set of predicted keypoints data
        # true_kp contains a set of true keypoints data

        # Reshape them to (22, 3)
        pred_kp_reshaped = pred_kp.reshape((22, 3))
        true_kp_reshaped = true_kp.reshape((22, 3))

        # Calculate the threshold for each keypoint
        thresholds = calculate_thresholds(true_kp_reshaped, phenotype_rules)

        # Calculate the pmp for each keypoint
        for i in range(num_keypoints):
            pmp_values[i] += np.linalg.norm(pred_kp_reshaped[i, :2] - true_kp_reshaped[i, :2]) <= thresholds[i] * t

    # Normalize pmp values
    pmp_values /= len(pred_keypoints)
    return pmp_values


def calculate_thresholds(keypoints, phenotype_rules):
    """
    Calculate the threshold for each keypoint.
    keypoints: Array of keypoints with shape (num_keypoints, 2 or 3).
    phenotype_rules: Dictionary of phenotype length calculation rules.
    """
    num_keypoints = keypoints.shape[0]
    thresholds = np.zeros(num_keypoints)

    for (kpt, kpt1_idx, kpt2_idx), calc_type in phenotype_rules.items():
        kpt1 = keypoints[kpt1_idx, :2]  # Consider only x, y coordinates
        kpt2 = keypoints[kpt2_idx, :2]

        if calc_type == 'x':
            length = abs(kpt1[0] - kpt2[0])
        elif calc_type == 'y':
            length = abs(kpt1[1] - kpt2[1])
        elif calc_type == 'z':
            length = np.linalg.norm(kpt1 - kpt2)
        else:
            raise ValueError("Unknown calculation type: '{}'".format(calc_type))

        thresholds[kpt - 1] = length  # Subtract 1 from the index to match Python's 0-based indexing

    return thresholds


class EvalCOCOMetric:
    def __init__(self,
                 coco: COCO = None,
                 iou_type: str = "keypoints",
                 results_file_name: str = "predict_results.json",
                 classes_mapping: dict = None,
                 threshold: float = 0.2):
        self.coco = copy.deepcopy(coco)
        self.obj_ids = []  # Record the ids of the targets (persons) processed by each process
        self.results = []
        self.aggregation_results = None
        self.classes_mapping = classes_mapping
        self.coco_evaluator = None
        assert iou_type in ["keypoints"]
        self.iou_type = iou_type
        self.results_file_name = results_file_name
        self.threshold = threshold

    def plot_img(self, img_path, keypoints, r=3):
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)
        for i, point in enumerate(keypoints):
            draw.ellipse([point[0] - r, point[1] - r, point[0] + r, point[1] + r],
                         fill=(255, 0, 0))
        img.show()

    def prepare_for_coco_keypoints(self, targets, outputs):
        # Iterate through the prediction results of each person (note this is not each image, one image may have multiple persons)
        for target, keypoints, scores in zip(targets, outputs[0], outputs[1]):
            if len(keypoints) == 0:
                continue

            obj_idx = int(target["obj_index"])
            if obj_idx in self.obj_ids:
                # Prevent duplicate data
                continue

            self.obj_ids.append(obj_idx)
            # self.plot_img(target["image_path"], keypoints)

            mask = np.greater(scores, 0.2)
            if mask.sum() == 0:
                k_score = 0
            else:
                k_score = np.mean(scores[mask])

            keypoints = np.concatenate([keypoints, scores], axis=1)
            keypoints = np.reshape(keypoints, -1)

            # We recommend rounding coordinates to the nearest tenth of a pixel to reduce resulting JSON file size.
            keypoints = [round(k, 2) for k in keypoints.tolist()]

            res = {"image_id": target["image_id"],
                   "category_id": 1,  # person
                   "keypoints": keypoints,
                   "score": target["score"] * k_score}

            self.results.append(res)

    def update(self, targets, outputs):
        if self.iou_type == "keypoints":
            self.prepare_for_coco_keypoints(targets, outputs)
        else:
            raise KeyError(f"not support iou_type: {self.iou_type}")

    def synchronize_results(self):
        # Synchronize data across all processes
        eval_ids, eval_results = merge(self.obj_ids, self.results)
        self.aggregation_results = {"obj_ids": eval_ids, "results": eval_results}

        # Save only on the main process
        if is_main_process():
            # results = []
            # [results.extend(i) for i in eval_results]
            # Write predict results into json file
            json_str = json.dumps(eval_results, indent=4)
            with open(self.results_file_name, 'w') as json_file:
                json_file.write(json_str)

    def evaluate(self):
        # Evaluate only on the main process
        if is_main_process():
            # accumulate predictions from all images
            coco_true = self.coco
            coco_pre = coco_true.loadRes(self.results_file_name)
            num_keypoints = 22
            anns = [ann['id'] for ann in coco_pre.dataset['annotations']]
            # Use 'anns' to index into 'coco_pre'
            pred_keypoints = np.array([coco_pre.anns[id]['keypoints'] for id in anns])

            anns1 = [ann['id'] for ann in coco_true.dataset['annotations']]
            # Use 'anns' to index into 'coco_true'
            true_keypoints = np.array([coco_true.anns[id]['keypoints'] for id in anns1])

            phenotype_rules = {
                (1, 0, 10): 'x',  # Keypoint index starts from 0, so 1 is 0, 11 is 10 1-11 snout length
                (2, 1, 11): 'x',  # 2-12 postorbital length
                (3, 2, 3): 'y',  # 3-4 head height
                (4, 2, 3): 'y',  # 3-4 head height
                (5, 4, 5): 'y',  # 5-6 body height
                (6, 4, 5): 'y',  # 5-6 body height
                (7, 6, 7): 'y',  # 7-8 caudal peduncle height
                (8, 6, 7): 'y',  # 7-8 caudal peduncle height
                (9, 8, 9): 'x',  # 9-10 caudal fin length
                (10, 8, 9): 'x',  # 9-10 caudal fin length
                (11, 10, 11): 'x',  # 11-12 eye diameter
                (12, 10, 11): 'x',  # 11-12 eye diameter
                (13, 12, 13): 'z',  # 13-14 pectoral fin length
                (14, 12, 13): 'z',  # 13-14 pectoral fin length
                (15, 14, 15): 'z',  # 15-16 pelvic fin length
                (16, 14, 15): 'z',  # 15-16 pelvic fin length
                (17, 16, 18): 'z',  # 17-19 anal fin length
                (18, 16, 17): 'z',  # 17-18 anal fin base length
                (19, 16, 18): 'z',  # 17-19 anal fin length
                (20, 19, 20): 'z',  # 20-21 dorsal fin base length
                (21, 19, 20): 'z',  # 20-21 dorsal fin base length
                (22, 19, 21): 'z'  # 20-22 dorsal fin length
                # ... Add other phenotype length calculation rules
            }
            # Define keypoint names
            keypoint_names = [
                "Snout Tip-1", "Posterior Edge of Operculum-2", "Highest Point of Head-3", "Isthmus-4",
                "Highest Point of Dorsal Edge-5", "Lowest Point of Ventral Edge-6", "Upper Caudal Peduncle-7",
                "Lower Caudal Peduncle-8", "End of Caudal Fin-9", "Posterior Edge of Caudal Vertebra-10",
                "Anterior Edge of Eye-11", "Posterior Edge of Eye-12", "Starting Point of Pectoral Fin-13",
                "End of Pectoral Fin Base-14", "Starting Point of Pelvic Fin-15", "End of Pelvic Fin Base-16",
                "Starting Point of Anal Fin-17", "Posterior Edge of Anal Fin Base-18", "Outer Edge of Anal Fin-19",
                "Starting Point of Dorsal Fin-20",
                "Posterior Edge of Dorsal Fin Base-21", "Outer Edge of Dorsal Fin-22"
            ]
            # Set different thresholds
            t_values = [0.05, 0.1, 0.2, 0.5]
            print('---------------------------pmp---------------------------------------------')
            # Define an empty dictionary to store information
            results = []
            # Loop through each threshold
            for t in t_values:
                # Call the function to calculate pmp values
                pmp_values = calculate_pmp(pred_keypoints, true_keypoints, phenotype_rules, num_keypoints, t)

                # Calculate the average of pmp_values
                average_pmp = np.mean(pmp_values)

                # Print the current threshold
                print('---------------------------')
                print(f"pmp threshold t = {t}")
                print('---------------------------')

                # Create a temporary dictionary to store the current threshold and the pmp values and average for each keypoint
                temp_result = {"pmp_threshold": t, "keypoints_pmp": {}, "average_pmp": average_pmp}

                # Print the pmp value for each keypoint
                for name, pmp in zip(keypoint_names, pmp_values):
                    print(f"{name}: {pmp}")
                    temp_result["keypoints_pmp"][name] = pmp

                results.append(temp_result)

            # self.coco_evaluator = COCOeval(cocoGt=coco_true, cocoDt=coco_pre, iouType=self.iou_type)

            # self.coco_evaluator.evaluate()
            # self.coco_evaluator.accumulate()
            # print(f"IoU metric: {self.iou_type}")
            # self.coco_evaluator.summarize()

            # coco_info = self.coco_evaluator.stats.tolist()  # numpy to list
            return results
        else:
            return None
