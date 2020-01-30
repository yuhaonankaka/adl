import argparse
import os

import numpy as np
from torchvision1.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision1.models.detection.transform import GeneralizedRCNNTransform
from tqdm import tqdm
from PIL import Image
import torchvision1.transforms.functional as TF
import torch

def get_model(num_classes=18):
    model = resnet_fpn_backbone('resnet18', True)
    return model

def store_features_for_projection_multi(imagePaths, model_maskrcnn, outputs,sceneid):
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    # these transform parameters are from source code of Mask R CNN
    transform = GeneralizedRCNNTransform(min_size=800, max_size=1333, image_mean=image_mean, image_std=image_std)
    images = [Image.open(imagePath) for imagePath in imagePaths]
    image_tensors = [TF.to_tensor(image) for image in images]
    # let it be in list (can be multiple)
    images, _ = transform(image_tensors)
    with torch.no_grad():
        body = model_maskrcnn.body
        body = body
        output = body(images.tensors)
    torch.save(output, outputs+sceneid+".fea")


def transfer(key):
    return int(key)

def store_2d_features(args):
    save_path = args.output
    with open('../data/ScanRefer_filtered_all.txt', 'r') as f:
        scan_names = f.read().splitlines()
    for scene_id in tqdm(scan_names):
        image_path = os.path.join('/home/davech2y/frames_square', scene_id, 'color')
        images = []
        for image_name in os.listdir(image_path):
            image_name = image_name.replace(".jpg", "", 1)
            images.append(image_name)
        images.sort(key = transfer)
        interval = round(len(images) / 40)
        if interval == 0:
            interval = 1
        indices = np.arange(0, len(images), interval)
        indices = indices[:40]
        indices = indices.astype('int32')
        indices = list(indices)
        data_path = os.path.join('/home/davech2y/frames_square',scene_id)
        images = [images[i] for i in indices]
        maskrcnn_model = get_model()
        store_features_for_projection_multi([os.path.join(data_path, 'color', image_name + ".jpg") for image_name in images], maskrcnn_model, save_path, scene_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, help="output folder", default="")
    args = parser.parse_args()
    store_2d_features(args)