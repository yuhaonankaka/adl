import warnings

import torch
import torch.nn as nn
import numpy as np
import sys
import os

from utils.projection import Projection

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from models.backbone_module import Pointnet2Backbone
from models.voting_module import VotingModule
from models.ref_module import RefModule


class RefNet(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, input_feature_dim=0, num_proposal=128, vote_factor=1, sampling="vote_fps", use_lang_classifier=True, num_nearest_images=0):
        super().__init__()
        self.num_nearest_images = num_nearest_images
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
        self.pooling = nn.MaxPool1d(kernel_size=num_nearest_images)

        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, 256)

        # Vote aggregation, detection and language reference
        self.rfnet = RefModule(num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling, use_lang_classifier)

    def forward(self, data_dict, imageft, proj_ind_3d, proj_ind_2d, num_points):
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

        batch_size = data_dict["point_clouds"].shape[0]

        # Back project imageft to 3d space and concat with point_clouds
        num_images = proj_ind_3d.shape[0] // batch_size
        imageft_back3d = [Projection.apply(ft, ind3d, ind2d, num_points)
                          for ft, ind3d, ind2d in zip(imageft, proj_ind_3d, proj_ind_2d)]
        imageft_back3d = torch.stack(imageft_back3d, dim=2)  # shape: (n_ft_channels, n_sampled_pts, batch_size*n_img)

        # Max Pool
        if num_images == self.num_nearest_images:
            imageft_back3d = self.pooling(imageft_back3d)  # shape: (n_ft_channels, n_sampled_pts, batch_size)
        else:
            warnings.warn("votenet.py: num_images != self.num_images")
            imageft_back3d = nn.MaxPool1d(kernel_size=num_images)(imageft_back3d)

        # Rearrange the dims
        imageft_back3d = imageft_back3d.permute(2, 1, 0)

        # Directly use the aligned pcl to concat features, because we already have the mappings of indices.
        # Alignment operation does not change indices, but only the (x,y,z) value.
        pcl_enriched = torch.cat((data_dict['point_clouds'], imageft_back3d), dim=2)
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
