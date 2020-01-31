from tqdm import tqdm
import os
import math
import imageio
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.autograd import Function

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

class ProjectionHelper():
    def __init__(self, intrinsic, depth_min, depth_max, image_dims):
        self.intrinsic = intrinsic
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.image_dims = image_dims


    def depth_to_skeleton(self, ux, uy, depth):
        x = (ux - self.intrinsic[0][2]) / self.intrinsic[0][0]
        y = (uy - self.intrinsic[1][2]) / self.intrinsic[1][1]
        return torch.Tensor([depth*x, depth*y, depth])


    def skeleton_to_depth(self, p):
        x = (p[0] * self.intrinsic[0][0]) / p[2] + self.intrinsic[0][2]
        y = (p[1] * self.intrinsic[1][1]) / p[2] + self.intrinsic[1][2]
        return torch.Tensor([x, y, p[2]])

    def compute_frustum_bounds(self, camera_to_world):
        corner_points = camera_to_world.new(8, 4, 1).fill_(1)
        # depth min
        corner_points[0][:3] = self.depth_to_skeleton(0, 0, self.depth_min).unsqueeze(1)
        corner_points[1][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, 0, self.depth_min).unsqueeze(1)
        corner_points[2][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, self.image_dims[1] - 1,
                                                      self.depth_min).unsqueeze(1)
        corner_points[3][:3] = self.depth_to_skeleton(0, self.image_dims[1] - 1, self.depth_min).unsqueeze(1)
        # depth max
        corner_points[4][:3] = self.depth_to_skeleton(0, 0, self.depth_max).unsqueeze(1)
        corner_points[5][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, 0, self.depth_max).unsqueeze(1)
        corner_points[6][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, self.image_dims[1] - 1,
                                                      self.depth_max).unsqueeze(1)
        corner_points[7][:3] = self.depth_to_skeleton(0, self.image_dims[1] - 1, self.depth_max).unsqueeze(1)

        p = torch.bmm(camera_to_world.repeat(8, 1, 1), corner_points)
        # pl = torch.round(torch.bmm(world_to_grid.repeat(8, 1, 1), torch.floor(p)))
        # pu = torch.round(torch.bmm(world_to_grid.repeat(8, 1, 1), torch.ceil(p)))

        p = p.squeeze()
        p = p.cpu().numpy()
        p = torch.from_numpy(p)
        p = torch.unsqueeze(p, 2)

        bbox_min0, _ = torch.min(p[:, :3, 0], 0)
        # bbox_min1, _ = torch.min(pu[:, :3, 0], 0)
        # bbox_min = np.minimum(bbox_min0, bbox_min1)
        bbox_max0, _ = torch.max(p[:, :3, 0], 0)
        # bbox_max1, _ = torch.max(pu[:, :3, 0], 0)
        # bbox_max = np.maximum(bbox_max0, bbox_max1)
        return bbox_min0, bbox_max0

        # TODO make runnable on cpu as well...

    def compute_frustum_bounds_multi(self, camera_to_worlds):
        n_images = camera_to_worlds.shape[0]
        corner_points = camera_to_worlds.new(n_images, 8, 4, 1).fill_(1)
        # depth min
        corner_points[:n_images,0,:3] = self.depth_to_skeleton(0, 0, self.depth_min).unsqueeze(1).repeat(n_images,1,1)
        corner_points[:n_images,1,:3] = self.depth_to_skeleton(self.image_dims[0] - 1, 0, self.depth_min).unsqueeze(1).repeat(n_images,1,1)
        corner_points[:n_images,2,:3] = self.depth_to_skeleton(self.image_dims[0] - 1, self.image_dims[1] - 1, self.depth_min).unsqueeze(1).repeat(n_images,1,1)
        corner_points[:n_images,3,:3] = self.depth_to_skeleton(0, self.image_dims[1] - 1, self.depth_min).unsqueeze(1).repeat(n_images,1,1)
        # depth max
        corner_points[:n_images,4,:3] = self.depth_to_skeleton(0, 0, self.depth_max).unsqueeze(1).repeat(n_images,1,1)
        corner_points[:n_images,5,:3] = self.depth_to_skeleton(self.image_dims[0] - 1, 0, self.depth_max).unsqueeze(1).repeat(n_images,1,1)
        corner_points[:n_images,6,:3] = self.depth_to_skeleton(self.image_dims[0] - 1, self.image_dims[1] - 1,self.depth_max).unsqueeze(1).repeat(n_images,1,1)
        corner_points[:n_images,7,:3] = self.depth_to_skeleton(0, self.image_dims[1] - 1, self.depth_max).unsqueeze(1).repeat(n_images,1,1)

        camera_to_worlds_tiled = np.tile(camera_to_worlds.cpu().numpy().reshape(-1,16),8).reshape(n_images,8,4,4)
        camera_to_worlds_tiled = torch.from_numpy(camera_to_worlds_tiled)
        p = torch.bmm(camera_to_worlds_tiled.view(-1,4,4), corner_points.view(-1,4,1))
        p = p.view(n_images,8,4,1)
        # pl = torch.round(torch.bmm(world_to_grid.repeat(8, 1, 1), torch.floor(p)))
        # pu = torch.round(torch.bmm(world_to_grid.repeat(8, 1, 1), torch.ceil(p)))

        p = p.squeeze()
        p = p.cpu().numpy()
        p = torch.from_numpy(p)
        p = torch.unsqueeze(p, 3)

        bbox_min0, _ = torch.min(p[:,:, :3, 0], 1)
        # bbox_min1, _ = torch.min(pu[:, :3, 0], 0)
        # bbox_min = np.minimum(bbox_min0, bbox_min1)
        bbox_max0, _ = torch.max(p[:,:, :3, 0], 1)
        # bbox_max1, _ = torch.max(pu[:, :3, 0], 0)
        # bbox_max = np.maximum(bbox_max0, bbox_max1)
        return bbox_min0, bbox_max0

    def compute_projection_multi(self, camera_to_worlds):
        # compute projection by voxels -> image
        world_to_camera = torch.inverse(camera_to_worlds)
        # voxel_bounds_min, voxel_bounds_max = self.compute_frustum_bounds_multi(camera_to_worlds)
        # return voxel_bounds_min, voxel_bounds_max, world_to_camera
        return world_to_camera

    def compute_projection(self, camera_to_world):
        # compute projection by voxels -> image
        world_to_camera = torch.inverse(camera_to_world)
        voxel_bounds_min, voxel_bounds_max = self.compute_frustum_bounds(camera_to_world)
        return voxel_bounds_min, voxel_bounds_max, world_to_camera


        # voxel_bounds_min = np.maximum(voxel_bounds_min, 0).cuda()
        # voxel_bounds_max = np.minimum(voxel_bounds_max, self.volume_dims).float().cuda()
        #
        # # coordinates within frustum bounds
        # lin_ind_volume = torch.arange(0, self.volume_dims[0]*self.volume_dims[1]*self.volume_dims[2], out=torch.LongTensor()).cuda()
        # coords = camera_to_world.new(4, lin_ind_volume.size(0))
        # coords[2] = lin_ind_volume / (self.volume_dims[0]*self.volume_dims[1])
        # tmp = lin_ind_volume - (coords[2]*self.volume_dims[0]*self.volume_dims[1]).long()
        # coords[1] = tmp / self.volume_dims[0]
        # coords[0] = torch.remainder(tmp, self.volume_dims[0])
        # coords[3].fill_(1)
        # mask_frustum_bounds = torch.ge(coords[0], voxel_bounds_min[0]) * torch.ge(coords[1], voxel_bounds_min[1]) * torch.ge(coords[2], voxel_bounds_min[2])
        # mask_frustum_bounds = mask_frustum_bounds * torch.lt(coords[0], voxel_bounds_max[0]) * torch.lt(coords[1], voxel_bounds_max[1]) * torch.lt(coords[2], voxel_bounds_max[2])
        # if not mask_frustum_bounds.any():
        #     #print('error: nothing in frustum bounds')
        #     return None
        # lin_ind_volume = lin_ind_volume[mask_frustum_bounds]
        # coords = coords.resize_(4, lin_ind_volume.size(0))
        # coords[2] = lin_ind_volume / (self.volume_dims[0]*self.volume_dims[1])
        # tmp = lin_ind_volume - (coords[2]*self.volume_dims[0]*self.volume_dims[1]).long()
        # coords[1] = tmp / self.volume_dims[0]
        # coords[0] = torch.remainder(tmp, self.volume_dims[0])
        # coords[3].fill_(1)
        #
        # # transform to current frame
        # p = torch.mm(world_to_camera, torch.mm(grid_to_world, coords))
        #
        # # project into image
        # p[0] = (p[0] * self.intrinsic[0][0]) / p[2] + self.intrinsic[0][2]
        # p[1] = (p[1] * self.intrinsic[1][1]) / p[2] + self.intrinsic[1][2]
        # pi = torch.round(p).long()
        #
        # valid_ind_mask = torch.ge(pi[0], 0) * torch.ge(pi[1], 0) * torch.lt(pi[0], self.image_dims[0]) * torch.lt(pi[1], self.image_dims[1])
        # if not valid_ind_mask.any():
        #     #print('error: no valid image indices')
        #     return None
        # valid_image_ind_x = pi[0][valid_ind_mask]
        # valid_image_ind_y = pi[1][valid_ind_mask]
        # valid_image_ind_lin = valid_image_ind_y * self.image_dims[0] + valid_image_ind_x
        # depth_vals = torch.index_select(depth.view(-1), 0, valid_image_ind_lin)
        # depth_mask = depth_vals.ge(self.depth_min) * depth_vals.le(self.depth_max) * torch.abs(depth_vals - p[2][valid_ind_mask]).le(self.voxel_size)
        #
        # if not depth_mask.any():
        #     #print('error: no valid depths')
        #     return None
        #
        # lin_ind_update = lin_ind_volume[valid_ind_mask]
        # lin_ind_update = lin_ind_update[depth_mask]
        # lin_indices_3d = lin_ind_update.new(self.volume_dims[0]*self.volume_dims[1]*self.volume_dims[2] + 1) #needs to be same size for all in batch... (first element has size)
        # lin_indices_2d = lin_ind_update.new(self.volume_dims[0]*self.volume_dims[1]*self.volume_dims[2] + 1) #needs to be same size for all in batch... (first element has size)
        # lin_indices_3d[0] = lin_ind_update.shape[0]
        # lin_indices_2d[0] = lin_ind_update.shape[0]
        # lin_indices_3d[1:1+lin_indices_3d[0]] = lin_ind_update
        # lin_indices_2d[1:1+lin_indices_2d[0]] = torch.index_select(valid_image_ind_lin, 0, torch.nonzero(depth_mask)[:,0])
        # num_ind = lin_indices_3d[0]
        #
        # return lin_indices_3d, lin_indices_2d

# Inherit from Function
class Projection(Function):

    @staticmethod
    def forward(ctx, label, lin_indices_3d, lin_indices_2d, volume_dims):
        ctx.save_for_backward(lin_indices_3d, lin_indices_2d)
        num_label_ft = 1 if len(label.shape) == 2 else label.shape[0]
        output = label.new(num_label_ft, volume_dims[2], volume_dims[1], volume_dims[0]).fill_(0)
        num_ind = lin_indices_3d[0]
        if num_ind > 0:
            vals = torch.index_select(label.view(num_label_ft, -1), 1, lin_indices_2d[1:1+num_ind])
            output.view(num_label_ft, -1)[:, lin_indices_3d[1:1+num_ind]] = vals
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_label = grad_output.clone()
        num_ft = grad_output.shape[0]
        grad_label.data.resize_(num_ft, 32, 41)
        lin_indices_3d, lin_indices_2d = ctx.saved_variables
        num_ind = lin_indices_3d.data[0]
        vals = torch.index_select(grad_output.data.contiguous().view(num_ft, -1), 1, lin_indices_3d.data[1:1+num_ind])
        grad_label.data.view(num_ft, -1)[:, lin_indices_2d.data[1:1+num_ind]] = vals
        return grad_label, None, None, None




def read_lines_from_file(filename):
    assert os.path.isfile(filename)
    lines = open(filename).read().splitlines()
    return lines


# create camera intrinsics
def make_intrinsic(fx, fy, mx, my):
    intrinsic = np.eye(4)
    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = mx
    intrinsic[1][2] = my
    return intrinsic


# create camera intrinsics
def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0,0] *= float(resize_width)/float(intrinsic_image_dim[0])
    intrinsic[1,1] *= float(image_dim[1])/float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0,2] *= float(image_dim[0]-1)/float(intrinsic_image_dim[0]-1)
    intrinsic[1,2] *= float(image_dim[1]-1)/float(intrinsic_image_dim[1]-1)
    return intrinsic


def load_pose(filename):
    pose = torch.Tensor(4, 4)
    lines = open(filename).read().splitlines()
    assert len(lines) == 4
    lines = [[x[0],x[1],x[2],x[3]] for x in (x.split(" ") for x in lines)]
    return torch.from_numpy(np.asarray(lines).astype(np.float32))


def resize_crop_image(image, new_image_dims):
    image_dims = [image.shape[1], image.shape[0]]
    if image_dims == new_image_dims:
        return image
    resize_width = int(math.floor(new_image_dims[1] * float(image_dims[0]) / float(image_dims[1])))
    image = transforms.Resize([new_image_dims[1], resize_width], interpolation=Image.NEAREST)(Image.fromarray(image))
    image = transforms.CenterCrop([new_image_dims[1], new_image_dims[0]])(image)
    image = np.array(image)
    return image


def load_depth_label_pose(depth_file, color_file, pose_file, depth_image_dims, color_image_dims, normalize):
    color_image = imageio.imread(color_file)
    depth_image = imageio.imread(depth_file)
    pose = load_pose(pose_file)
    # preprocess
    depth_image = resize_crop_image(depth_image, depth_image_dims)
    color_image = resize_crop_image(color_image, color_image_dims)
    depth_image = depth_image.astype(np.float32) / 1000.0
    color_image =  np.transpose(color_image, [2, 0, 1])  # move feature to front
    color_image = normalize(torch.Tensor(color_image.astype(np.float32) / 255.0))
    return depth_image, color_image, pose

def load_frames_multi(data_path, batch_scan_names, num_images, depth_images, color_images, poses, color_mean, color_std, choice='even'):

    depth_files = []
    color_files = []
    pose_files = []

    for scan_name in batch_scan_names:
        depth_path = os.path.join(data_path, scan_name, "depth")
        color_path = os.path.join(data_path, scan_name, "color")
        pose_path = os.path.join(data_path, scan_name, "pose")

        img_list = np.array(sorted(os.listdir(os.path.join(color_path))))
        depth_list = np.array(sorted(os.listdir(os.path.join(depth_path))))
        pose_list = np.array(sorted(os.listdir(os.path.join(pose_path))))
        assert len(img_list) == len(depth_list) == len(pose_list), "Data in %r have inconsistent amount of files" % data_path

        indices = np.array([])
        if choice == 'random':
            # Choose <num_images> frames randomly
            if len(img_list) < num_images:
                remain = num_images
                while remain > len(img_list):
                    indices = np.concatenate((indices, np.arange(len(img_list), dtype='int32')))
                    remain -= len(img_list)
                indices = np.concatenate((indices.astype('int32'), np.arange(remain, dtype='int32')))
            else:
                indices = np.sort(np.random.choice(len(img_list), num_images, replace=False)).astype('int32')
        elif choice == 'even':
            # Choose <num_images> frames according to the total number of frames. For example:
            # if `total_num_frames=20`, and `num_images=6`--> `interval=round(20/6)=3`
            # then the indices of the chosen frames are: [0,3,6,9,12,15]
            if len(img_list) < num_images:
                while len(indices) < num_images:
                    indices = np.append(indices, np.arange(len(img_list)))
                indices = indices[:num_images].astype('int32')
            else:
                interval = round(len(img_list) / num_images)
                if interval * (num_images - 1) > len(img_list) - 1:  # just in case, not really necessary
                    interval -= 1
                indices = np.arange(0, len(img_list), interval)
                indices = indices[:num_images]
                indices = indices.astype('int32')
        else:
            Exception("choice='{}' is not valid, please choose from ['random', 'even']".format(choice))


        for idx in indices:
            color_files.append(os.path.join(data_path, scan_name, 'color', img_list[idx]))
            depth_files.append(os.path.join(data_path, scan_name, 'depth', depth_list[idx]))
            pose_files.append(os.path.join(data_path, scan_name, 'pose', pose_list[idx]))

        # color_files.append(img_list[indices])
        # depth_files.append(depth_list[indices])
        # pose_files.append(pose_list[indices])

    depth_image_dims = [depth_images.shape[2], depth_images.shape[1]]
    color_image_dims = [color_images.shape[3], color_images.shape[2]]
    normalize = transforms.Normalize(mean=color_mean, std=color_std)

    # load data
    for k in range(len(batch_scan_names) * num_images):
        depth_image, color_image, pose = load_depth_label_pose(depth_files[k], color_files[k], pose_files[k],
                                                               depth_image_dims, color_image_dims, normalize)
        color_images[k] = color_image
        depth_images[k] = torch.from_numpy(depth_image)
        poses[k] = pose




# def load_frames_multi(data_path, frame_indices, depth_images, color_images, poses, color_mean, color_std):
#     # construct files
#     num_images = frame_indices.shape[1] - 2
#     scan_names = ['scene' + str(scene_id).zfill(4) + '_' + str(scan_id).zfill(2) for scene_id, scan_id in frame_indices[:,:2].numpy()]
#     scan_names = np.repeat(scan_names, num_images)
#     frame_ids = frame_indices[:, 2:].contiguous().view(-1).numpy()
#     depth_files = [os.path.join(data_path, scan_name, 'depth', str(frame_id) + '.png') for scan_name, frame_id in zip(scan_names, frame_ids)]
#     color_files = [os.path.join(data_path, scan_name, 'color', str(frame_id) + '.jpg') for scan_name, frame_id in zip(scan_names, frame_ids)]
#     pose_files = [os.path.join(data_path, scan_name, 'pose', str(frame_id) + '.txt') for scan_name, frame_id in zip(scan_names, frame_ids)]
#
#     batch_size = frame_indices.size(0) * num_images
#     depth_image_dims = [depth_images.shape[2], depth_images.shape[1]]
#     color_image_dims = [color_images.shape[3], color_images.shape[2]]
#     normalize = transforms.Normalize(mean=color_mean, std=color_std)
#     # load data
#     for k in range(batch_size):
#         depth_image, color_image, pose = load_depth_label_pose(depth_files[k], color_files[k], pose_files[k], depth_image_dims, color_image_dims, normalize)
#         color_images[k] = color_image
#         depth_images[k] = torch.from_numpy(depth_image)
#         poses[k] = pose

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






