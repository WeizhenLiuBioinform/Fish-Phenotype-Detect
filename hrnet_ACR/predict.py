import os
import json

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from model import HighResolutionNet
# from draw_utils import draw_keypoints
import transforms


def draw_keypoints(image, keypoints, scores, thresh=0.5, r=5, color=(0, 255, 0), font_scale=2, font_thickness=3):
    # def draw_keypoints(image, keypoints, scores, thresh=0.5, r=0.3, color=(0, 255, 0), font_scale=0.5, font_thickness=1):
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


def predict_single_person():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    flip_test = False
    # resize_hw = (1728,2304)
    resize_hw = (3456, 4608)
    # resize_hw = (6912,9216)
    # resize_hw = (864,1152)
    # img_path = "/home/tanjy/data/data json/shibanyu_json_706/val/2208030001_20220803161528.jpg"
    # img_path = "/home/tanjy/data/data_200/sby/val/2108110012_20210811170957.jpg"
    # img_path = "/home/tanjy/data/data_200/sby/train/2108110011_20210811150443.jpg"
    img_path = "/home/tanjy/data/sby_50/val/2206230001_20220623153206.jpg"
    # weights_path = "/home/tanjy/code/HRnet/save_weights/model-61.pth"

    # weights_path = "/home/tanjy/code/HRnet/save_weights/200/hbjl_test2/best_model.pth"
    # weights_path = "/home/tanjy/code/HRnet/save_weights/200/hbjl/model-190.pth"
    # weights_path = "/home/tanjy/code/HRnet/save_weights/200/hbjl/model-100.pth"

    # weights_path = "/home/tanjy/code/HRnet/save_weights/200/sby/model-190.pth"
    # weights_path = "/home/tanjy/code/HRnet/save_weights/200/sby_ok/best_model.pth"

    # weights_path = "/home/tanjy/code/HRnet/save_weights/200/pty/model-190.pth"
    # weights_path = "/home/tanjy/code/HRnet/save_weights/200/pty/model-80.pth"
    # weights_path = "/home/tanjy/code/HRnet/save_weights/200/pty/model-160.pth"
    weights_path = "/home/tanjy/code/HRnet_copy/save_weights/sby_50/sby_3/best_model.pth"

    keypoint_json_path = "/home/tanjy/code/HRnet/person_keypoints.json"
    assert os.path.exists(img_path), f"file: {img_path} does not exist."
    assert os.path.exists(weights_path), f"file: {weights_path} does not exist."
    assert os.path.exists(keypoint_json_path), f"file: {keypoint_json_path} does not exist."

    data_transform = transforms.Compose([
        # transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=resize_hw),
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
    # img_tensor, target = data_transform(img, {"box": [0, 0, img.shape[1] - 1, img.shape[0] - 1]})
    img_tensor, target = data_transform(img, {"box": [23, 2418, 4572, 1416]})
    img_tensor = torch.unsqueeze(img_tensor, dim=0)

    # create model
    # HRNet-W32: base_channel=32
    # HRNet-W48: base_channel=48
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
            # feature is not aligned, shift flipped heatmap for higher accuracy
            # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/22
            flip_outputs[..., 1:] = flip_outputs.clone()[..., 0: -1]
            outputs = (outputs + flip_outputs) * 0.5

        keypoints, scores = transforms.get_final_preds(outputs, [target["reverse_trans"]], True)
        keypoints = np.squeeze(keypoints)
        scores = np.squeeze(scores)
        print(keypoints)
        print(keypoints.shape)
        print(len(keypoints))

        plot_img = draw_keypoints(img, keypoints, scores, thresh=0.005, r=15)
        # plot_img = draw_keypoints(img, keypoints, scores, thresh=0.005, r=2)
        plt.imshow(plot_img)
        plt.show()
        # Save the image using cv2.imwrite
        # cv2.imwrite("predict_sby_test.jpg", cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite("predict_sby_test.jpg", cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR))
        # plot_img.save("test_result13.jpg")


if __name__ == '__main__':
    from PIL import Image

    # Read image file

    predict_single_person()
