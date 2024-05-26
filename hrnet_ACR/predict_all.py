import os
import json

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from model import HighResolutionNet
# from draw_utils import draw_keypoints
import transforms


def predict_all_person():
    # Placeholder for future implementation
    pass


import cv2
import numpy as np


def draw_keypoints(image, keypoints, scores, thresh=0.5, r=0.3, color=(0, 255, 0), font_scale=0.5, font_thickness=1):
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


def predict_single_person(img_path, output_folder):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    flip_test = False
    resize_hw = (864, 1152)

    weights_path = "/home/tanjy/code/HRnet_copy/save_weights/pty_200_new/pty_5/best_model.pth"
    keypoint_json_path = "/home/tanjy/code/HRnet_copy/person_keypoints.json"

    assert os.path.exists(img_path), f"file: {img_path} does not exist."
    assert os.path.exists(weights_path), f"file: {weights_path} does not exist."
    assert os.path.exists(keypoint_json_path), f"file: {keypoint_json_path} does not exist."

    data_transform = transforms.Compose([
        transforms.AffineTransform(scale=None, fixed_size=resize_hw),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # read json file
    with open(keypoint_json_path, "r") as f:
        person_info = json.load(f)

    # read single-person image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor, target, extra_info = data_transform(img, {"box": [0, 0, img.shape[1] - 1, img.shape[0] - 1]})
    img_tensor = torch.unsqueeze(img_tensor, dim=0)

    # create model
    model = HighResolutionNet(base_channel=48, num_joints=22)
    weights = torch.load(weights_path, map_location=device)
    weights = weights if "model" not in weights else weights["model"]
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    with torch.inference_mode():
        outputs = model(img_tensor.to(device))

        if flip_test:
            flip_tensor = transforms.flip_images(img_tensor)
            flip_outputs = torch.squeeze(
                transforms.flip_back(model(flip_tensor.to(device)), person_info["flip_pairs"]),
            )
            flip_outputs[..., 1:] = flip_outputs.clone()[..., 0: -1]
            outputs = (outputs + flip_outputs) * 0.5

        keypoints, scores = transforms.get_final_preds(outputs, [target["reverse_trans"]], True)
        keypoints = np.squeeze(keypoints)
        scores = np.squeeze(scores)
        print(keypoints)
        print(len(keypoints))

        plot_img = draw_keypoints(img, keypoints, scores, thresh=0.005, r=2)
        plt.imshow(plot_img)
        plt.show()

        output_file = os.path.join(output_folder, os.path.basename(img_path))
        cv2.imwrite(output_file, cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR))


def predict_all_person(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            predict_single_person(img_path, output_folder)


if __name__ == '__main__':
    input_folder = "/home/tanjy/data/new_data_200/pty/val"
    output_folder = "/home/tanjy/data/predict_data/predict_200new/predict_200_pty_5"
    predict_all_person(input_folder, output_folder)
