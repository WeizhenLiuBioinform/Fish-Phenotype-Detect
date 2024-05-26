from my_dataset_coco import CocoKeypoint
import transforms
import numpy as np
import json
from torch.utils import data
from model import HighResolutionNet
import os
import json
from scipy.stats import pearsonr
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt


def calculate_mape(true_values, predicted_values):
    """
    Calculate MAPE for a single instance.

    Args:
    - true_values (list): List of true values.
    - predicted_values (list): List of predicted values.

    Returns:
    - mape (float): MAPE value.
    """
    true_values, predicted_values = np.array(true_values), np.array(predicted_values)
    if np.any(true_values == 0):
        return None  # Return None if any true value is 0, as MAPE cannot be calculated
    return np.mean(np.abs((true_values - predicted_values) / true_values)) * 100


def calculate_ddef_for_single_instance(keypoints, visible, phenotype_rules):
    """
    Calculate phenotype lengths for a single instance based on the given rules.

    Args:
    - keypoints (list): List of keypoints for a single instance, each keypoint is [x, y, visibility].
    - visible (list): List of visibility for each keypoint.
    - phenotype_rules (dict): Dictionary of phenotype length calculation rules, where keys are pairs of keypoint indices and values are calculation types ('x', 'y', or 'z').

    Returns:
    - d_def_values (list): List of phenotype lengths for each keypoint.
    """
    d_def_values = [0] * 23
    phenotype_index = 0

    for (kpt, kpt1_idx, kpt2_idx), calc_type in phenotype_rules.items():
        # Check if both keypoints are visible (visibility value equals 2)
        if visible[kpt1_idx] == 2 and visible[kpt2_idx] == 2:
            kpt1 = keypoints[kpt1_idx]
            kpt2 = keypoints[kpt2_idx]

            if calc_type == 'x':
                length = abs(kpt1[0] - kpt2[0])
            elif calc_type == 'y':
                length = abs(kpt1[1] - kpt2[1])
            elif calc_type == 'z':
                length = np.linalg.norm(np.array(kpt1) - np.array(kpt2))
            else:
                raise ValueError("Unknown calculation type: '{}'".format(calc_type))

            d_def_values[phenotype_index] += length

        phenotype_index += 1
        if phenotype_index >= 23:
            break

    return d_def_values


# Assume your COCO dataset is stored at this path
# coco_dataset_path = "/home/tanjy/data/new_data_200/hbjl_new"
# coco_dataset_path = "/home/tanjy/data/new_data_200/pty"
# coco_dataset_path = "/home/tanjy/data/new_data_200/sby_new"
# coco_dataset_path = "/home/tanjy/data/new_data_200/sby_new_new"
coco_dataset_path = "/home/tanjy/data/new_data_200/hbjl_new_new"

with open("/home/tanjy/code/HRnet_copy/person_keypoints.json", "r") as f:
    person_kps_info = json.load(f)
fixed_size = [1728, 2304]
# fixed_size=[432,576]
# fixed_size=[3456,4608]
# fixed_size = [864,1152]
# fixed_size = [1152,864]
fixed_size = fixed_size
heatmap_hw = (fixed_size[0] // 4, fixed_size[1] // 4)
kps_weights = np.array(person_kps_info["kps_weights"], dtype=np.float32).reshape((22,))
data_transform = {
    "train": transforms.Compose([
        # transforms.HalfBody(0.3, person_kps_info["upper_body_ids"], person_kps_info["lower_body_ids"]),
        # transforms.AffineTransform(scale=(0.65, 1.35), rotation=(-45, 45), fixed_size=fixed_size),
        transforms.AffineTransform(scale=None, rotation=None, fixed_size=fixed_size),
        # transforms.RandomHorizontalFlip(0.5, person_kps_info["flip_pairs"]),
        transforms.KeypointToHeatMap(heatmap_hw=heatmap_hw, gaussian_sigma=2, keypoints_weights=kps_weights),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        # transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=fixed_size),
        transforms.AffineTransform(scale=None, rotation=None, fixed_size=fixed_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# Create CocoKeypoint dataset instance
coco_dataset = CocoKeypoint(root=coco_dataset_path, dataset="val", transforms=data_transform["val"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# weights_path = "/home/tanjy/code/HRnet_copy/save_weights/hbjl_200_new/hbjl_3/best_model.pth"
# weights_path = "/home/tanjy/code/HRnet_copy/save_weights/sby_200_new/sby_6/best_model.pth"
# weights_path = "/home/tanjy/code/HRnet/save_weights/pty_200_new/pty_5/best_model.pth"
weights_path = "/home/tanjy/code/HRnet_copy/save_weights/hbjl_200_new_new/hbjl_2/best_model.pth"
model = HighResolutionNet(base_channel=48, num_joints=22)
weights = torch.load(weights_path, map_location=device)
weights = weights if "model" not in weights else weights["model"]
model.load_state_dict(weights)
model.to(device)
model.eval()
# Use DataLoader for batching and shuffling
from torch.utils.data import DataLoader

train_data_loader = data.DataLoader(coco_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    pin_memory=True,
                                    num_workers=0,
                                    collate_fn=coco_dataset.collate_fn)

# phenotype_rules = {
#     (1,0, 10): 'x',  # Keypoint indices start from 0, so 1 is 0, 11 is 10 1-11 Snout length
#     (2,1, 11): 'x',  # 2-12 Postorbital head length
#     (3,2, 3): 'y',   # 3-4 Head height
#     (4,2, 3): 'y',   # 3-4 Head height
#     (5,4, 5): 'y',   # 5-6 Body height
#     (6,4, 5): 'y',   # 5-6 Body height
#     (7,6, 7): 'y',   # 7-8 Caudal peduncle height
#     (8,6, 7): 'y',   # 7-8 Caudal peduncle height
#     (9,8, 9): 'x',  # 9-10 Caudal fin length
#     (10,8, 9): 'x',  # 9-10 Caudal fin length
#     (11,10, 11): 'x', # 11-12 Eye diameter
#     (12,10, 11): 'x', # 11-12 Eye diameter
#     (13,12, 13): 'z', # 13-14 Pectoral fin length
#     (14,12, 13): 'z', # 13-14 Pectoral fin length
#     (15,14, 15): 'z', # 15-16 Pelvic fin length
#     (16,14, 15): 'z', # 15-16 Pelvic fin length
#     (17,16, 18): 'z', # 17-19 Anal fin length
#     (18,16, 17): 'z', # 17-18 Anal fin base length
#     (19,16, 18): 'z', # 17-19 Anal fin length
#     (20,19, 20): 'z', # 20-21 Dorsal fin base length
#     (21,19, 20): 'z', # 20-21 Dorsal fin base length
#     (22,19, 21): 'z'  # 20-22 Dorsal fin length
#     # ... Add other phenotype length calculation rules
# }
phenotype_rules = {
    (1, 0, 8): 'x',  # Keypoint indices start from 0, so 1 is 0, 9 is 8 1-9 Total length
    (2, 0, 9): 'x',  # 1-10 Body length
    (3, 0, 1): 'x',  # 1-2 Head length
    (4, 0, 10): 'x',  # 1-11 Snout length
    (5, 10, 11): 'x',  # 11-12 Eye diameter
    (6, 1, 11): 'x',  # 2-12 Postorbital head length
    (7, 4, 5): 'y',  # 5-6 Body height
    (8, 2, 3): 'y',  # 3-4 Head height
    (9, 14, 16): 'z',  # 15-17 Abdominal anal fin base distance
    (10, 6, 7): 'y',  # 7-8 Caudal peduncle height
    (11, 9, 17): 'x',  # 10-18 Caudal peduncle length
    (12, 19, 20): 'z',  # 20-21 Dorsal fin base length
    (13, 12, 13): 'z',  # 13-14 Pectoral fin length
    (14, 14, 15): 'z',  # 15-16 Pelvic fin length
    (15, 16, 17): 'z',  # 17-18 Anal fin base length
    (16, 8, 9): 'x',  # 9-10 Caudal fin length
    (17, 0, 19): 'x',  # 1-20 Pre-dorsal fin distance
    (18, 9, 19): 'x',  # 10-20 Post-dorsal fin distance
    (19, 12, 19): 'z',  # 13-20 Pectoral dorsal fin distance
    (20, 12, 14): 'z',  # 13-15 Pectoral pelvic fin distance
    (21, 14, 19): 'z',  # 15-20 Pelvic dorsal fin distance
    (22, 16, 18): 'z',  # 17-19 Anal fin length
    (23, 19, 21): 'z'  # 20-22 Dorsal fin length
    # ... Add other phenotype length calculation rules
}
phenotype_names = [
    "Total length-1", "Body length-2", "Head length-3", "Snout length-4", " Eye diameter-5",
    "Postorbital head length-6", "Body height-7", "Head height-8", "Abdominal anal fin base distance-9",
    "Caudal peduncle height-10",
    "Caudal peduncle length-11", "Dorsal fin base length-12", "Pectoral fin length-13", "Pelvic fin length-14",
    "Anal fin base length-15", "Caudal fin length-16", " Pre-dorsal fin distance-17", "Post-dorsal fin distance-18",
    " Pectoral dorsal fin distance-19", "Pectoral pelvic fin distance-20",
    "Pelvic dorsal fin distance-21", "Anal fin length-22", "Dorsal fin length-23"
]
# Load COCO formatted JSON file
# with open("/home/tanjy/data/new_data_200/sby_new_new/annotations/val.json", "r") as f:
# with open("/home/tanjy/data/new_data_200/hbjl_new/annotations/val.json", "r") as f:
# with open("/home/tanjy/data/new_data_200/pty/annotations/val.json", "r") as f:
with open("/home/tanjy/data/new_data_200/hbjl_new_new/annotations/val.json", "r") as f:
    coco_data = json.load(f)
# Create a map from image_id to its keypoint data
keypoints_map = {ann['image_id']: ann['keypoints'] for ann in coco_data['annotations']}

# Initialize accumulators
phenotype_mapes_accumulator = {i: [] for i in range(23)}

# Initialize accumulators to store true and predicted values (Pearson correlation)
phenotype_true_values_accumulator = {i: [] for i in range(23)}
phenotype_predicted_values_accumulator = {i: [] for i in range(23)}

# Iterate over the dataset
for images, targets, scale, angle, src_center in train_data_loader:
    # Get the image ID
    image_id = targets[0]["image_id"]
    # Get the original keypoints from the map
    true_keypoints = keypoints_map[image_id]
    true_keypoints = np.array(true_keypoints).reshape((-1, 3))  # COCO format is usually [x, y, v]

    # Extract coordinates and visibility
    keypoints_coords = true_keypoints[:, :2]
    # visible = true_keypoints[:, 2]
    true_keypoints = np.array(keypoints_coords).reshape((22, 2))
    visible = np.array(targets[0]["visible"])

    # Calculate d_def values for a single instance
    true_d_def_values = calculate_ddef_for_single_instance(true_keypoints, visible, phenotype_rules)

    # Get the predicted values
    with torch.inference_mode():
        outputs = model(images.to(device))
    reverse_trans = [t["reverse_trans"] for t in targets]
    keypoints, scores = transforms.get_final_preds(outputs, reverse_trans, True)
    keypoints = np.squeeze(keypoints)
    # scores = np.squeeze(scores)
    print(keypoints)
    print(true_keypoints)
    # print(len(keypoints))
    predicted_d_def_values = calculate_ddef_for_single_instance(keypoints, visible, phenotype_rules)
    # Calculate and accumulate MAPE for each phenotype
    for idx in range(23):
        mape = calculate_mape([true_d_def_values[idx]], [predicted_d_def_values[idx]])
        if mape is not None:
            phenotype_mapes_accumulator[idx].append(mape)

    print("hhhhhhhhhh")

    # Accumulate true and predicted values
    for idx in range(23):
        phenotype_true_values_accumulator[idx].append(true_d_def_values[idx])
        phenotype_predicted_values_accumulator[idx].append(predicted_d_def_values[idx])

# Calculate the average MAPE for each phenotype
average_phenotype_mapes = {idx: np.mean(mapes) if mapes else None for idx, mapes in phenotype_mapes_accumulator.items()}

# Output the average MAPE for each phenotype
for idx, mape in average_phenotype_mapes.items():
    phenotype_name = phenotype_names[idx] if idx < len(phenotype_names) else f"Phenotype {idx + 1}"
    print(f"Average MAPE for {phenotype_name}: {mape if mape is not None else 'N/A'}%")

# Calculate the overall average MAPE (only including phenotypes that can be calculated)
overall_mape = np.mean([mape for mape in average_phenotype_mapes.values() if mape is not None])
print(f"Overall Average MAPE: {overall_mape}%")

import pandas as pd

# Prepare data
data = {
    "Phenotype": [],
    "Average MAPE (%)": []
}

for idx, mape in average_phenotype_mapes.items():
    phenotype_name = phenotype_names[idx] if idx < len(phenotype_names) else f"Phenotype {idx + 1}"
    data["Phenotype"].append(phenotype_name)
    data["Average MAPE (%)"].append(mape if mape is not None else 'N/A')

# Convert data to DataFrame
df = pd.DataFrame(data)

# Save DataFrame to CSV file
csv_file_path = "/home/tanjy/code/HRnet_copy/phenotype_csv/mape_csv/hbjl_new_phenotype_mapes_2.csv"  # You can change the file path and name
df.to_csv(csv_file_path, index=False)

# Calculate and output Pearson correlation coefficients and p-values for each phenotype
pearson_correlations = {}

for idx in range(23):
    true_values = phenotype_true_values_accumulator[idx]
    predicted_values = phenotype_predicted_values_accumulator[idx]

    # Ensure the lists are not empty and have the same length
    if true_values and predicted_values and len(true_values) == len(predicted_values):
        correlation, p_value = pearsonr(true_values, predicted_values)
        pearson_correlations[idx] = (correlation, p_value)
    else:
        pearson_correlations[idx] = (None, None)

# Output Pearson correlation coefficients and p-values for each phenotype
for idx, (correlation, p_value) in pearson_correlations.items():
    phenotype_name = phenotype_names[idx] if idx < len(phenotype_names) else f"Phenotype {idx + 1}"
    print(
        f"Pearson Correlation for {phenotype_name}: Correlation = {correlation if correlation is not None else 'N/A'}, p-value = {p_value if p_value is not None else 'N/A'}")

# Save Pearson correlation results to CSV
import pandas as pd

# Prepare data
data = {
    "Phenotype": [],
    "Pearson Correlation": [],
    "p-value": []
}

for idx, (correlation, p_value) in pearson_correlations.items():
    phenotype_name = phenotype_names[idx] if idx < len(phenotype_names) else f"Phenotype {idx + 1}"
    data["Phenotype"].append(phenotype_name)
    data["Pearson Correlation"].append(correlation if correlation is not None else 'N/A')
    data["p-value"].append(p_value if p_value is not None else 'N/A')

# Convert data to DataFrame
df = pd.DataFrame(data)

# Save DataFrame to CSV file
csv_file_path = "/home/tanjy/code/HRnet_copy/phenotype_csv/pearson_csv/hbjl_new_pearson_csv_2.csv"  # Replace with your desired path and filename
df.to_csv(csv_file_path, index=False)

# Plot R-squared
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


# Prepare a function to calculate R-squared
def calculate_r_squared(x, y):
    model = LinearRegression()
    model.fit(x[:, np.newaxis], y)
    return model.score(x[:, np.newaxis], y)


# Prepare to plot and calculate R-squared
r_squared_values = {}

for idx in range(23):
    true_values = np.array(phenotype_true_values_accumulator[idx])
    predicted_values = np.array(phenotype_predicted_values_accumulator[idx])

    if len(true_values) > 1 and len(predicted_values) > 1:
        # Calculate R-squared
        r_squared = calculate_r_squared(true_values, predicted_values)
        r_squared_values[idx] = r_squared

        # Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(true_values, predicted_values, color='blue')
        plt.title(f'Phenotype {phenotype_names[idx]}: Regression Fit')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')

        # Add fit line
        m, b = np.polyfit(true_values, predicted_values, 1)
        plt.plot(true_values, m * true_values + b, color='red')

        # plt.savefig(f'/home/tanjy/code/HRnet_copy/phenotype_csv/image/image_6/fit_plot_{idx}.png')  # Save image
        plt.savefig(
            f'/home/tanjy/code/HRnet_copy/phenotype_csv/image/hbjl_new_image_2/fit_plot_{idx}.png')  # Save image
        plt.close()

# Output R-squared values
for idx, r_squared in r_squared_values.items():
    print(f"Phenotype {phenotype_names[idx]}: R-squared = {r_squared}")

# Optionally save R-squared values to CSV
r_squared_data = {
    "Phenotype": [phenotype_names[idx] for idx in r_squared_values],
    "R-squared": [r_squared_values[idx] for idx in r_squared_values]
}

df_r_squared = pd.DataFrame(r_squared_data)
csv_r_squared_path = "/home/tanjy/code/HRnet_copy/phenotype_csv/r_csv/hbjl_new_file_r_squared_2.csv"  # Replace with your desired path and filename
df_r_squared.to_csv(csv_r_squared_path, index=False)
