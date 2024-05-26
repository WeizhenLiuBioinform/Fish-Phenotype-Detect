import numpy as np
from torch.utils.data import DataLoader
from my_dataset_coco import CocoKeypoint
import transforms
import numpy as np
import json
from torch.utils import data
import cv2
import math


def apply_reverse_transform(keypoints, reverse_trans):
    transformed_keypoints = []
    for kp in keypoints:
        # Multiply keypoints by the reverse transformation matrix
        kp_homogeneous = np.array([kp[0], kp[1], 1.0])  # Convert to homogeneous coordinates
        transformed_kp = np.dot(reverse_trans, kp_homogeneous)
        transformed_keypoints.append(transformed_kp[:2])

    return np.array(transformed_keypoints)


# Assuming your COCO dataset path is the same as before
# coco_dataset_path = "/home/tanjy/data/new_data_200/sby_new"
# coco_dataset_path = "/home/tanjy/data/new_data_200/sby_new_new"
# coco_dataset_path = "/home/tanjy/data/new_data_200/hbjl_new"
coco_dataset_path = "/home/tanjy/data/new_data_200/hbjl_new_new"
# coco_dataset_path = "/home/tanjy/data/new_data_200/pty"

# Create transformation functions
data_transforms = transforms.Compose([

    # transforms.AffineTransform(scale=None, rotation=None, fixed_size=[432,576]),
    # transforms.AffineTransform(scale=None, rotation=None, fixed_size=[216,288]),
    transforms.AffineTransform(scale=None, rotation=None, fixed_size=[3456, 4608]),
    # transforms.AffineTransform(scale=(0.65, 1.35), rotation=(-45, 45), fixed_size=[3456,4608]),
    transforms.ToTensor()
    # Other required transformations
])

# Create CocoKeypoint dataset instances for training and validation sets
train_dataset = CocoKeypoint(root=coco_dataset_path, dataset="train", transforms=data_transforms)
val_dataset = CocoKeypoint(root=coco_dataset_path, dataset="val", transforms=data_transforms)

# Use DataLoader for batching and shuffling data
train_data_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=0,
                                    collate_fn=train_dataset.collate_fn)
val_data_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=0,
                                  collate_fn=val_dataset.collate_fn)


def normalize_keypoints(keypoints):
    """Normalize keypoints for a single data instance"""
    # Find min and max values for x and y coordinates
    min_x, max_x = np.min(keypoints[:, 0]), np.max(keypoints[:, 0])
    min_y, max_y = np.min(keypoints[:, 1]), np.max(keypoints[:, 1])

    # Ensure calculations are done using float
    min_x, max_x = float(min_x), float(max_x)
    min_y, max_y = float(min_y), float(max_y)
    keypoints = keypoints.astype(float)

    # Normalize x and y coordinates
    keypoints[:, 0] = (keypoints[:, 0] - min_x) / (max_x - min_x) if max_x != min_x else 0.5
    keypoints[:, 1] = (keypoints[:, 1] - min_y) / (max_y - min_y) if max_y != min_y else 0.5

    return keypoints


def update_global_stats(keypoints, global_max_min):
    """Update global max-min statistics"""
    for idx, (x, y) in enumerate(keypoints):
        global_max_min[idx]['max_x'] = max(global_max_min[idx]['max_x'], x)
        global_max_min[idx]['max_y'] = max(global_max_min[idx]['max_y'], y)
        global_max_min[idx]['min_x'] = min(global_max_min[idx]['min_x'], x)
        global_max_min[idx]['min_y'] = min(global_max_min[idx]['min_y'], y)
    return global_max_min


def process_data_loaders(data_loaders, num_keypoints):
    global_max_min = {i: {'max_x': float('-inf'), 'min_x': float('inf'), 'max_y': float('-inf'), 'min_y': float('inf')}
                      for i in range(num_keypoints)}

    for data_loader in data_loaders:
        for images, targets, scale, angle, src_center in data_loader:
            transform_params = {'scale': scale, 'angle': angle, 'dst_center': src_center}
            keypoints = np.array(targets[0]["keypoints"]).reshape((-1, 2))  # Assuming each keypoint is in (x, y) format

            reverse_trans = targets[0]["reverse_trans"]
            # Reverse transform keypoints
            transformed_keypoints = apply_reverse_transform(keypoints, reverse_trans)

            # Normalize keypoint coordinates
            keypoints = normalize_keypoints(transformed_keypoints)

            global_max_min = update_global_stats(keypoints, global_max_min)

    return global_max_min


num_keypoints = 22  # Assuming there are 22 keypoints

# Process all data loaders
global_max_min = process_data_loaders([train_data_loader, val_data_loader], num_keypoints)

# Output globally normalized max and min coordinates
print("Global Normalized Max and Min Coordinates:", global_max_min)

import csv


def save_to_csv(data, filename):
    """Save data to a CSV file"""
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write header row
        writer.writerow(['Keypoint', 'Max_X', 'Min_X', 'Max_Y', 'Min_Y'])

        for keypoint, coords in data.items():
            writer.writerow([keypoint, coords['max_x'], coords['min_x'], coords['max_y'], coords['min_y']])


# Assuming global_max_min is the dictionary computed earlier
# Call the function to save it as a CSV file
save_to_csv(global_max_min, 'hbjl_global_max_min_coordinates_fanal.csv')
