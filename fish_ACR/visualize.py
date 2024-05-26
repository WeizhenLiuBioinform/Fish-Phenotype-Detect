from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import os
import cv2

def draw_keypoints(image, keypoints, r=2, color=(255, 165, 0)):
    """
    Draw keypoints on the image.
    :param image: The image on which to draw keypoints.
    :param keypoints: List of keypoints, each keypoint in (x, y, v) format, where v is the visibility flag.
    :param r: Radius of the keypoints.
    :param color: Color of the keypoints.
    """
    for i in range(0, len(keypoints), 3):
        x, y, v = keypoints[i], keypoints[i+1], keypoints[i+2]
        if v > 0:  # v>0 indicates the keypoint is visible
            cv2.circle(image, (int(x), int(y)), r, color, -1)
    return image

def visualize_annotations(coco_annotation_file, image_dir, output_dir):
    coco = COCO(coco_annotation_file)
    img_ids = coco.getImgIds()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        image_path = os.path.join(image_dir, img_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            image = draw_keypoints(image, ann['keypoints'])

        output_path = os.path.join(output_dir, img_info['file_name'])
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

# Call the function
# coco_annotation_file = '/home/tanjy/data/new_data_200/sby/annotations/val.json'
# coco_annotation_file = '/home/tanjy/data/new_data_200/hbjl/annotations/val.json'
coco_annotation_file = '/home/tanjy/data/new_data_200/pty/annotations/val.json'
# coco_annotation_file = '/home/tanjy/data/sby_50/annotations/val.json'
# coco_annotation_file = '/home/tanjy/data/new_data_200/hbjl_new/annotations/val.json'
# coco_annotation_file = '/home/tanjy/data/new_data_200/sby_new_new/annotations/val.json'
# image_dir = '/home/tanjy/data/predict_data/predict_sby'
# image_dir = '/home/tanjy/data/predict_data/predictsby_200/predicthbjl_200_11_1'
# image_dir = '/home/tanjy/data/predict_data/predict_200new/predict_200_hbjl_3'
image_dir = '/home/tanjy/data/predict_data/predict_200new/predict_200_pty_6'
# output_dir = '/home/tanjy/data/predict_data/v_sby'
# output_dir = '/home/tanjy/data/predict_data/v/v_hbjl200_11'
# output_dir = '/home/tanjy/data/predict_data/v/v_hbjl200new_3'
output_dir = '/home/tanjy/data/predict_data/v/v_pty200new_6'
visualize_annotations(coco_annotation_file, image_dir, output_dir)
