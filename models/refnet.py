import warnings

import torch
import torch.nn as nn
import numpy as np
import sys
import os

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from utils.projection import Projection
from utils.projection_until import scannet_projection

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from models.backbone_module import Pointnet2Backbone
from models.voting_module import VotingModule
from models.ref_module import RefModule
from utils import image_util
from utils.image_util import read_lines_from_file

ENET_TYPES = {'scannet': (41, [0.496342, 0.466664, 0.440796], [0.277856, 0.28623, 0.291129])}

input_image_dims = [320, 240]
# proj_image_dims = [40, 30]  # feature dimension of ENet
proj_image_dims = [34, 25]
# proj_image_dims = [320, 240]  # feature dimension of ENet
color_mean = [0.496342, 0.466664, 0.440796]
color_std = [0.277856, 0.28623, 0.291129]

def get_intrinsics(scene_id, args):
    intrinsic_str = read_lines_from_file(args.data_path_2d + '/' + scene_id + '/intrinsic_depth.txt')
    fx = float(intrinsic_str[0].split()[0])
    fy = float(intrinsic_str[1].split()[1])
    mx = float(intrinsic_str[0].split()[2])
    my = float(intrinsic_str[1].split()[2])
    intrinsic = image_util.make_intrinsic(fx, fy, mx, my)
    intrinsic = image_util.adjust_intrinsic(intrinsic, [args.intrinsic_image_width, args.intrinsic_image_height],
                                      proj_image_dims)
    return intrinsic


class RefNet(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, input_feature_dim=0, num_proposal=128, vote_factor=1, sampling="vote_fps", use_lang_classifier=True):
        super().__init__()
        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling=sampling
        self.use_lang_classifier=use_lang_classifier

        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, 256)

        # mask r cnn
        self.maskrcnn_model = resnet_fpn_backbone('resnet18', True).fpn.cuda()

        # Vote aggregation, detection and language reference
        self.rfnet = RefModule(num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling, use_lang_classifier)

    def forward(self, data_dict, args):
        """ Forward pass of the network

        Args:
            data_dict: dict
                {
                    point_clouds, 
                    lang_feat
                }

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """

        # =======================================
        # Get 3d <-> 2D Projection Mapping and 2D feature map
        # =======================================
        batch_size = len(data_dict['scan_name'])
        new_features = torch.zeros((batch_size, self.args.num_points, 32)).cuda()
        for idx, scene_id in enumerate(data_dict['scan_name']):
            intrinsics = get_intrinsics(scene_id, self.args)
            projection = Projection.ProjectionHelper(intrinsics, self.args.depth_min, self.args.depth_max, proj_image_dims)
            features_2d = scannet_projection(data_dict['point_clouds'][idx].cpu().numpy(), intrinsics, projection,
                                             scene_id, self.args, None, None, self.maskrcnn_model)
            new_features[idx, :] = features_2d[:]
        data_dict['new_features'] = new_features
        pcl_enriched = torch.cat((data_dict['point_clouds'], data_dict['new_features']), dim=2)
        data_dict['point_clouds'] = pcl_enriched


        data_dict = self.backbone_net(data_dict)
                
        # --------- HOUGH VOTING ---------
        xyz = data_dict["fp2_xyz"]
        features = data_dict["fp2_features"]
        data_dict["seed_inds"] = data_dict["fp2_inds"]
        data_dict["seed_xyz"] = xyz
        data_dict["seed_features"] = features
        
        xyz, features = self.vgen(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        data_dict["vote_xyz"] = xyz
        data_dict["vote_features"] = features

        data_dict = self.rfnet(xyz, features, data_dict)

        return data_dict
