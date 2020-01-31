import os
import torch
import numpy as np
import torchvision.transforms as transforms
from ..utils.image_util import load_depth_label_pose
from ..utils.projection import ProjectionHelper
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# train on the GPU or on the CPU, if a GPU is not available
if torch.cuda.is_available():
    print("Using GPU to train")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

ENET_TYPES = {'scannet': (41, [0.496342, 0.466664, 0.440796], [0.277856, 0.28623, 0.291129])}  #classes, color mean/std
color_mean = [0.496342, 0.466664, 0.440796]
color_std = [0.277856, 0.28623, 0.291129]
# create camera intrinsics
input_image_dims = [320, 240]
# enet
# proj_image_dims = [40, 30]
# mask r cnn
proj_image_dims = [34, 25]

data_path_2d = '/home/davech2y/frames_square'

def read_lines_from_file(filename):
    assert os.path.isfile(filename)
    lines = open(filename).read().splitlines()
    return lines


def load_frames_multi_2(data_path, image_names, depth_image_torch, color_image_torch, camera_pose, color_mean, color_std):
    color_images = []
    depth_images = []
    camera_poses = []
    for image_name in image_names:
        depth_file = os.path.join(data_path, 'depth', image_name+".png")
        color_file = os.path.join(data_path, 'color', image_name+".jpg")
        pose_file = os.path.join(data_path,  'pose', image_name+".txt")
        depth_image_dims = [depth_image_torch.shape[2], depth_image_torch.shape[1]]
        color_image_dims = [color_image_torch.shape[3], color_image_torch.shape[2]]
        normalize = transforms.Normalize(mean=color_mean, std=color_std)
        # load data
        depth_img, color_img, pose = load_depth_label_pose(depth_file, color_file, pose_file, depth_image_dims, color_image_dims, normalize)
        color_image = color_img
        depth_image = torch.from_numpy(depth_img)
        camera_pose = pose
        color_images.append(color_image)
        depth_images.append(depth_image)
        camera_poses.append(camera_pose)
    return color_images,depth_images,camera_poses


def project_images(data_path, image_names,scene_id):
    depth_image = torch.cuda.FloatTensor(len(image_names), proj_image_dims[1], proj_image_dims[0])
    color_image = torch.cuda.FloatTensor(len(image_names), 3, input_image_dims[1], input_image_dims[0])
    camera_pose = torch.cuda.FloatTensor(len(image_names), 4, 4)
    color_images,depth_images,camera_poses = load_frames_multi_2(data_path, image_names, depth_image, color_image, camera_pose, color_mean, color_std)
    dict_2d = {}
    dict_2d["color"] = color_images
    dict_2d["depth"] = depth_images
    dict_2d["camera"] = camera_poses
    torch.save(dict_2d, "/home/davech2y/adl/lib/data_2d/"+scene_id +".dictt")
def transfer(key):
    return int(key)

def get_intrinsics(scene_id):
    intrinsic_str = read_lines_from_file(data_path_2d + '/' + scene_id + '/intrinsic_depth.txt')
    fx = float(intrinsic_str[0].split()[0])
    fy = float(intrinsic_str[1].split()[1])
    mx = float(intrinsic_str[0].split()[2])
    my = float(intrinsic_str[1].split()[2])
    intrinsic = image_util.make_intrinsic(fx, fy, mx, my)
    intrinsic = image_util.adjust_intrinsic(intrinsic, [640, 480],
                                      proj_image_dims)
    return intrinsic

# ROOT_DIR:indoor-objects
def scannet_projection(scene_id):
    # load_images
    intrinsics = get_intrinsics(scene_id)
    projection = ProjectionHelper(intrinsics, 0.4, 4.0, proj_image_dims)
    image_path = os.path.join(data_path_2d, scene_id, 'color')
    images = []
    for image_name in os.listdir(image_path):
        image_name = image_name.replace(".jpg", "", 1)
        images.append(image_name)
    images.sort(key=transfer)
    interval = round(len(images) / 40)
    if interval == 0:
        interval = 1
    indices = np.arange(0, len(images), interval)
    indices = indices[:40]
    indices = indices.astype('int32')
    indices = list(indices)
    data_path = os.path.join(data_path_2d, scene_id)
    images = [images[i] for i in indices]
    project_images(data_path, images, scene_id)

if __name__ == "__main__":
    with open('../data/ScanRefer_filtered_all.txt', 'r') as f:
        scan_names = f.read().splitlines()
    for scene_id in tqdm(scan_names):
        scannet_projection(scene_id)






