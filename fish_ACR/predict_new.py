import os
import json

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from model import HighResolutionNet
import transforms
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


def predict_all_person():
    # Placeholder for future implementation
    pass


import cv2
import numpy as np


def draw_keypoints(image, keypoints, scores, thresh=0.5, r=5, color=(0, 255, 0), font_scale=2, font_thickness=3):
    """
    Draw keypoints and their indices on the image.
    :param image: Image to draw keypoints on.
    :param keypoints: Array of keypoints.
    :param scores: Confidence scores of the keypoints.
    :param thresh: Confidence threshold, only keypoints with confidence higher than this value will be drawn.
    :param r: Radius of the keypoints.
    :param color: Color of the keypoints.
    :param font_scale: Font size for keypoint indices.
    :param font_thickness: Thickness of the font for keypoint indices.
    :return: Image with keypoints and their indices drawn on.
    """
    for i, (point, score) in enumerate(zip(keypoints, scores)):
        if score > thresh:
            x, y = int(point[0]), int(point[1])
            cv2.circle(image, (x, y), r, color, -1)

            # Draw keypoint indices
            cv2.putText(image, str(i + 1), (x + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0),
                        font_thickness)

    return image


def predict_single_person(data_transform, data_path, weights_path, device, save_folder):
    # Ensure the save folder exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Create dataset
    val_dataset = CocoKeypoint(data_path, "val", transforms=data_transform["val"])
    val_dataset_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True,
                                                     collate_fn=val_dataset.collate_fn)
    # Load model
    model = HighResolutionNet(num_joints=22)
    model.load_state_dict(torch.load(weights_path, map_location=device)['model'])
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i, (images, targets) in enumerate(val_dataset_loader):
            images = images.to(device)

            # Model prediction
            outputs = model(images)

            # Decode keypoints
            reverse_trans = [t['reverse_trans'] for t in targets]
            keypoints, scores = transforms.get_final_preds(outputs, reverse_trans, post_processing=True)
            keypoints = np.squeeze(keypoints, axis=0)
            scores = np.squeeze(scores, axis=0)

            # Save the original image to a temporary file
            temp_img_path = os.path.join(save_folder, f"temp_{i}.jpg")
            img_numpy = images[0].cpu().numpy()
            img_numpy = np.transpose(img_numpy, (1, 2, 0))
            img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
            cv2.imwrite(temp_img_path, img_numpy)

            # Read the temporary image
            img_to_draw = cv2.imread(temp_img_path)

            # Draw keypoints
            plot_img = draw_keypoints(img_to_draw, keypoints, scores, thresh=0.005, r=15)

            # Save the result to the specified folder
            save_path = os.path.join(save_folder, f"predict_result_{i}.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR))


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = {
        "val": transforms.Compose([
            transforms.AffineTransform(scale=None, fixed_size=[1728, 2304]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    data_path = '/home/tanjy/data/sby_50'
    weights_path = '/home/tanjy/code/HRnet_copy/save_weights/sby_50/sby_1/best_model.pth'
    save_folder = '/home/tanjy/data/predict_data/predictsby_50'

    predict_single_person(data_transform, data_path, weights_path, device, save_folder)


if __name__ == '__main__':
    main()
