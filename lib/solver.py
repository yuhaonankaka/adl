'''
File Created: Monday, 25th November 2019 1:35:30 pm
Author: Dave Zhenyu Chen (zhenyu.chen@tum.de)
'''

import os
import sys
import time
import warnings

import torch
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

from models.enet import create_enet_for_3d
from utils import image_util
from utils.projection import ProjectionHelper

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from lib.config import CONF
from lib.loss_helper import get_loss
from utils.eta import decode_eta


ITER_REPORT_TEMPLATE = """
-------------------------------iter: [{epoch_id}: {iter_id}/{total_iter}]-------------------------------
[loss] train_loss: {train_loss}
[loss] train_ref_loss: {train_ref_loss}
[loss] train_lang_loss: {train_lang_loss}
[loss] train_objectness_loss: {train_objectness_loss}
[loss] train_vote_loss: {train_vote_loss}
[loss] train_box_loss: {train_box_loss}
[loss] train_lang_acc: {train_lang_acc}
[sco.] train_ref_acc: {train_ref_acc}
[sco.] train_obj_acc: {train_obj_acc}
[sco.] train_pos_ratio: {train_pos_ratio}, train_neg_ratio: {train_neg_ratio}
[sco.] train_iou_rate_0.25: {train_iou_rate_25}, train_iou_rate_0.5: {train_iou_rate_5}
[info] mean_fetch_time: {mean_fetch_time}s
[info] mean_forward_time: {mean_forward_time}s
[info] mean_backward_time: {mean_backward_time}s
[info] mean_eval_time: {mean_eval_time}s
[info] mean_iter_time: {mean_iter_time}s
[info] ETA: {eta_h}h {eta_m}m {eta_s}s
"""

EPOCH_REPORT_TEMPLATE = """
---------------------------------summary---------------------------------
[train] train_loss: {train_loss}
[train] train_ref_loss: {train_ref_loss}
[train] train_lang_loss: {train_lang_loss}
[train] train_objectness_loss: {train_objectness_loss}
[train] train_vote_loss: {train_vote_loss}
[train] train_box_loss: {train_box_loss}
[train] train_lang_acc: {train_lang_acc}
[train] train_ref_acc: {train_ref_acc}
[train] train_obj_acc: {train_obj_acc}
[train] train_pos_ratio: {train_pos_ratio}, train_neg_ratio: {train_neg_ratio}
[train] train_iou_rate_0.25: {train_iou_rate_25}, train_iou_rate_0.5: {train_iou_rate_5}
[val]   val_loss: {val_loss}
[val]   val_ref_loss: {val_ref_loss}
[val]   val_lang_loss: {val_lang_loss}
[val]   val_objectness_loss: {val_objectness_loss}
[val]   val_vote_loss: {val_vote_loss}
[val]   val_box_loss: {val_box_loss}
[val]   val_lang_acc: {val_lang_acc}
[val]   val_ref_acc: {val_ref_acc}
[val]   val_obj_acc: {val_obj_acc}
[val]   val_pos_ratio: {val_pos_ratio}, val_neg_ratio: {val_neg_ratio}
[val]   val_iou_rate_0.25: {val_iou_rate_25}, val_iou_rate_0.5: {val_iou_rate_5}
"""

BEST_REPORT_TEMPLATE = """
--------------------------------------best--------------------------------------
[best] epoch: {epoch}
[loss] loss: {loss}
[loss] ref_loss: {ref_loss}
[loss] lang_loss: {lang_loss}
[loss] objectness_loss: {objectness_loss}
[loss] vote_loss: {vote_loss}
[loss] box_loss: {box_loss}
[loss] lang_acc: {lang_acc}
[sco.] ref_acc: {ref_acc}
[sco.] obj_acc: {obj_acc}
[sco.] pos_ratio: {pos_ratio}, neg_ratio: {neg_ratio}
[sco.] iou_rate_0.25: {iou_rate_25}, iou_rate_0.5: {iou_rate_5}
"""
# ------------------------------------------------------------------------- GLOBAL CONFIG BEG

#                    classes, color mean/std
# ENET_TYPES = {'scannet': (18, [0.496342, 0.466664, 0.440796], [0.277856, 0.28623, 0.291129])}
ENET_TYPES = {'scannet': (41, [0.496342, 0.466664, 0.440796], [0.277856, 0.28623, 0.291129])}

input_image_dims = [320, 240]
proj_image_dims = [40, 30]  # feature dimension of ENet
# proj_image_dims = [41, 32]  # feature dimension of ENet
# proj_image_dims = [320, 240]  # feature dimension of ENet
color_mean = [0.496342, 0.466664, 0.440796]
color_std = [0.277856, 0.28623, 0.291129]


def get_batch_intrinsics(batch_scan_names, args):
    """ Read intrinsics from txt file
    :param scan_name:
    :return: numpy array of shape: [batch_size, 4, 4]
    """
    batch_intrinsics = []
    for scan_name in batch_scan_names:
        intrinsic_str = image_util.read_lines_from_file(
            args.data_path_2d + '/' + scan_name + '/intrinsic_depth.txt')
        fx = float(intrinsic_str[0].split()[0])
        fy = float(intrinsic_str[1].split()[1])
        mx = float(intrinsic_str[0].split()[2])
        my = float(intrinsic_str[1].split()[2])

        intrinsic = image_util.make_intrinsic(fx, fy, mx, my)
        intrinsic = image_util.adjust_intrinsic(intrinsic,
                                                [args.intrinsic_image_width, args.intrinsic_image_height],
                                                proj_image_dims)
        batch_intrinsics.append(np.array(intrinsic))

    return np.array(batch_intrinsics)


def get_random_frames(data_path_2d, scan_name, args):
    img_path = os.path.join(args.data_path_2d, scan_name, "color")
    img_list = list(sorted(os.listdir(os.path.join(img_path))))
    indices = np.random.choice(len(img_list), args.num_nearest_images, replace=False)
    chosen_imgs = img_list[indices]
    return chosen_imgs


# ------------------------------------------------------------------------- GLOBAL CONFIG END

def project_2d_features(batch_data_label, args, model2d_fixed, model2d_trainable):
    """
        Retrieve certain amount of images(NUM_IMAGES) corresponding to the scene and find the correspondence mapping
        between 3D and 2D points.

        Parameters
        ----------
        batch_data_label: dict
            got from dataloader.


        Returns:
            proj_ind_3d: indexes of 3D points, index 0 is the number of valid points
            proj_ind_2d: indexes of 2D points, correspondent to indexes in proj_ind_3d. index 0 is number of valid points.
            imageft: 2D feature maps extracted from given 2D CNN
    """

    batch_scan_names = batch_data_label["scan_name"]

    # Get camera intrinsics for each scene
    batch_intrinsics = get_batch_intrinsics(batch_scan_names, args)

    # Get 2d images and it's feature
    depth_images = torch.cuda.FloatTensor(args.batch_size * args.num_nearest_images, proj_image_dims[1], proj_image_dims[0])
    color_images = torch.cuda.FloatTensor(args.batch_size * args.num_nearest_images, 3, input_image_dims[1], input_image_dims[0])
    camera_poses = torch.cuda.FloatTensor(args.batch_size * args.num_nearest_images, 4, 4)
    label_images = torch.cuda.LongTensor(args.batch_size * args.num_nearest_images, proj_image_dims[1],
                                         proj_image_dims[0])  # for proxy loss

    image_util.load_frames_multi(args.data_path_2d, batch_scan_names, args.num_nearest_images,
                                 depth_images, color_images, camera_poses, color_mean, color_std, choice='even')

    # Convert aligned point cloud back to unaligned, so that we can do back-projection
    # using camera intrinsics & extrinsics
    batch_pcl_aligned = batch_data_label['point_clouds']
    batch_pcl_unaligned = []
    batch_scan_names = batch_data_label['scan_name']
    # find the align matrix according to scan_name
    batch_align_matrix = np.array([])
    for scan_name, pcl_aligned in zip(batch_scan_names, batch_pcl_aligned):
        # Load alignments
        lines = open(args.RAW_DATA_DIR + scan_name + "/" + scan_name + ".txt")
        for line in lines:
            if 'axisAlignment' in line:
                axis_align_matrix = [float(x) \
                                     for x in line.rstrip().strip('axisAlignment = ').split(' ')]
                break
        axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
        # inv_axis_align_matrix_T = torch.inverse(torch.FloatTensor(axis_align_matrix.T))
        batch_align_matrix = np.append(batch_align_matrix, axis_align_matrix)
        inv_axis_align_matrix_T = np.linalg.inv(axis_align_matrix.T)

        # Numpy version:
        # Unalign the Point Cloud (See load_scannet_data.py as reference)
        pts = np.ones((pcl_aligned.size()[0], 4))
        pts[:, 0:3] = pcl_aligned[:, 0:3].cpu().numpy()
        pcl = np.dot(pts, inv_axis_align_matrix_T)
        batch_pcl_unaligned.append(torch.from_numpy(pcl).float())

        # # TEST
        # from test_helper import test_unalign_pcl
        # test_unalign_pcl(pcl_aligned.cpu().numpy(), pcl, scan_name, axis_align_matrix)
        # # END of TEST

        # Torch version:
        # Unalign the Point Cloud (See load_scannet_data.py as reference)
        # pcl = torch.ones(pcl_aligned.size()[0], 4)
        # pcl[:, 0:3] = pcl_aligned[:, 0:3]
        # pcl = torch.mm(pcl, inv_axis_align_matrix_T)
        # batch_pcl_unaligned.append(pcl)

    batch_pcl_unaligned = torch.stack(batch_pcl_unaligned)

    # Compute 3d <-> 2d projection mapping for each scene in the batch
    proj_mapping_list = []
    img_count = 0
    for d_img, c_pose in zip(depth_images, camera_poses):
        # TODO: double-check the curr_idx_batch
        curr_idx_batch = img_count // args.num_nearest_images
        if curr_idx_batch >= len(batch_scan_names):
            break
        # TEST
        # curr_scan_name = batch_scan_names[batch_idx]
        # pcl_root = "/home/kloping/Documents/TUM/3D_object_localization/data/scannet_point_clouds/"
        # pcl_path = os.path.join(pcl_root, curr_scan_name, curr_scan_name + "_vh_clean_2.ply")
        # destination = os.path.join(BASE_DIR, "utils", "test", "orig_mesh.ply")
        # shutil.copyfile(pcl_path, destination)
        # END of TEST
        projection = ProjectionHelper(batch_intrinsics[curr_idx_batch], args.depth_min, args.depth_max,
                                      proj_image_dims,
                                      args.num_points)
        proj_mapping = projection.compute_projection(batch_pcl_unaligned[curr_idx_batch], d_img, c_pose)
        proj_mapping_list.append(proj_mapping)
        img_count += 1

    if None in proj_mapping_list:  # invalid sample
        # print '(invalid sample)'
        return None, None, None
    proj_mapping = list(zip(*proj_mapping_list))
    proj_ind_3d = torch.stack(proj_mapping[0])
    proj_ind_2d = torch.stack(proj_mapping[1])

    # TODO: finish proxy loss part
    # if FLAGS.use_proxy_loss:
    #     data_util.load_label_frames(opt.data_path_2d, frames[v], label_images, num_classes)
    #     mask2d = label_images.view(-1).clone()
    #     for k in range(num_classes):
    #         if criterion_weights[k] == 0:
    #             mask2d[mask2d.eq(k)] = 0
    #     mask2d = mask2d.nonzero().squeeze()
    #     if (len(mask2d.shape) == 0):
    #         continue  # nothing to optimize for here

    # 2d features
    imageft_fixed = model2d_fixed(torch.autograd.Variable(color_images))
    imageft = model2d_trainable(imageft_fixed)
    # TODO: finish proxy loss part
    # if opt.use_proxy_loss:
    #     ft2d = model2d_classifier(imageft)
    #     ft2d = ft2d.permute(0, 2, 3, 1).contiguous()

    return proj_ind_3d, proj_ind_2d, imageft

class Solver():
    def __init__(self, model, config, dataloader, optimizer, stamp, val_step=10, use_lang_classifier=True, use_max_iou=False, args = {}):
        self.args = args
        self.epoch = 0                    # set in __call__
        self.verbose = 0                  # set in __call__
        
        self.model = model
        self.config = config
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.stamp = stamp
        self.val_step = val_step
        self.use_lang_classifier = use_lang_classifier
        self.use_max_iou = use_max_iou

        self.best = {
            "epoch": 0,
            "loss": float("inf"),
            "ref_loss": float("inf"),
            "lang_loss": float("inf"),
            "objectness_loss": float("inf"),
            "vote_loss": float("inf"),
            "box_loss": float("inf"),
            "lang_acc": -float("inf"),
            "ref_acc": -float("inf"),
            "obj_acc": -float("inf"),
            "pos_ratio": -float("inf"),
            "neg_ratio": -float("inf"),
            "iou_rate_0.25": -float("inf"),
            "iou_rate_0.5": -float("inf")
        }

        # log
        # contains all necessary info for all phases
        self.log = {
            phase: {
                # info
                "forward": [],
                "backward": [],
                "eval": [],
                "fetch": [],
                "iter_time": [],
                # loss (float, not torch.cuda.FloatTensor)
                "loss": [],
                "ref_loss": [],
                "lang_loss": [],
                "objectness_loss": [],
                "vote_loss": [],
                "box_loss": [],
                # scores (float, not torch.cuda.FloatTensor)
                "lang_acc": [],
                "ref_acc": [],
                "obj_acc": [],
                "pos_ratio": [],
                "neg_ratio": [],
                "iou_rate_0.25": [],
                "iou_rate_0.5": []
            } for phase in ["train", "val"]
        }
        
        # tensorboard
        os.makedirs(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/train"), exist_ok=True)
        os.makedirs(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/val"), exist_ok=True)
        self._log_writer = {
            "train": SummaryWriter(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/train")),
            "val": SummaryWriter(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/val"))
        }

        # training log
        log_path = os.path.join(CONF.PATH.OUTPUT, stamp, "log.txt")
        self.log_fout = open(log_path, "a")

        # private
        # only for internal access and temporary results
        self._running_log = {}
        self._global_iter_id = 0
        self._total_iter = {}             # set in __call__

        # templates
        self.__iter_report_template = ITER_REPORT_TEMPLATE
        self.__epoch_report_template = EPOCH_REPORT_TEMPLATE
        self.__best_report_template = BEST_REPORT_TEMPLATE
        # create model
        # model2d_fixed, model2d_trainable, model2d_classifier = create_enet_for_3d(ENET_TYPES['scannet'],
        #                                                                           FLAGS.model2d_path, DATASET_CONFIG.num_class)
        self.model2d_fixed, self.model2d_trainable, self.model2d_classifier = create_enet_for_3d(ENET_TYPES['scannet'],
                                                                                  self.args.model2d_path, 18)

        # move to gpu
        self.model2d_fixed = self.model2d_fixed.cuda()
        self.model2d_fixed.eval()
        self.model2d_trainable = self.model2d_trainable.cuda()
        self.model2d_classifier = self.model2d_classifier.cuda()

    def __call__(self, epoch, verbose):
        # setting
        self.epoch = epoch
        self.verbose = verbose
        self._total_iter["train"] = len(self.dataloader["train"]) * epoch
        self._total_iter["val"] = len(self.dataloader["val"]) * self.val_step
        
        for epoch_id in range(epoch):
            try:
                self._log("epoch {} starting...".format(epoch_id + 1))

                # feed 
                self._feed(self.dataloader["train"], "train", epoch_id)

                # save model
                self._log("saving last models...\n")
                model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
                torch.save(self.model.state_dict(), os.path.join(model_root, "model_last.pth"))
                
            except KeyboardInterrupt:
                # finish training
                self._finish(epoch_id)
                exit()

        # finish training
        self._finish(epoch_id)

    def _log(self, info_str):
        self.log_fout.write(info_str + "\n")
        self.log_fout.flush()
        print(info_str)

    def _set_phase(self, phase):
        if phase == "train":
            self.model.train()
        elif phase == "val":
            self.model.eval()
        else:
            raise ValueError("invalid phase")

    def _forward(self, data_dict, imageft, proj_ind_3d, proj_ind_2d, num_points):
        data_dict = self.model(data_dict, imageft, proj_ind_3d, proj_ind_2d, num_points)

        return data_dict

    def _backward(self):
        # optimize
        self.optimizer.zero_grad()
        self._running_log["loss"].backward()
        self.optimizer.step()

    def _compute_loss(self, data_dict):
        _, data_dict = get_loss(data_dict, self.config, True, self.use_lang_classifier, self.use_max_iou)

        # dump
        self._running_log["ref_loss"] = data_dict["ref_loss"]
        self._running_log["lang_loss"] = data_dict["lang_loss"]
        self._running_log["objectness_loss"] = data_dict["objectness_loss"]
        self._running_log["vote_loss"] = data_dict["vote_loss"]
        self._running_log["box_loss"] = data_dict["box_loss"]
        self._running_log["loss"] = data_dict["loss"]
      
    def _feed(self, dataloader, phase, epoch_id):
        # switch mode
        self._set_phase(phase)

        # change dataloader
        dataloader = dataloader if phase == "train" else tqdm(dataloader)

        for data_dict in dataloader:
            # move to cuda
            for key in data_dict:
                if key!='scan_name':
                    data_dict[key] = data_dict[key].cuda()

            # =======================================
            # Get 3d <-> 2D Projection Mapping and 2D feature map
            # =======================================
            proj_ind_3d, proj_ind_2d, imageft = project_2d_features(data_dict, self.args, self.model2d_fixed,
                                                                    self.model2d_trainable)
            if proj_ind_3d is None or proj_ind_2d is None or imageft is None:
                warnings.warn("Current training script: Projection invalid with scans: {}".format(data_dict['scan_name']))
                continue

            # initialize the running loss
            self._running_log = {
                # loss
                "loss": 0,
                "ref_loss": 0,
                "lang_loss": 0,
                "objectness_loss": 0,
                "vote_loss": 0,
                "box_loss": 0,
                # acc
                "lang_acc": 0,
                "ref_acc": 0,
                "obj_acc": 0,
                "pos_ratio": 0,
                "neg_ratio": 0,
                "iou_rate_0.25": 0,
                "iou_rate_0.5": 0
            }

            # load
            self.log[phase]["fetch"].append(data_dict["load_time"].sum().item())

            with torch.autograd.set_detect_anomaly(False):
                # forward
                start = time.time()
                data_dict = self._forward(data_dict, imageft, proj_ind_3d, proj_ind_2d, self.args.num_points)
                self._compute_loss(data_dict)
                self.log[phase]["forward"].append(time.time() - start)

                # backward
                if phase == "train":
                    start = time.time()
                    self._backward()
                    self.log[phase]["backward"].append(time.time() - start)
            
            # eval
            start = time.time()
            self._eval(data_dict)
            self.log[phase]["eval"].append(time.time() - start)

            # record log
            self.log[phase]["loss"].append(self._running_log["loss"].item())
            self.log[phase]["ref_loss"].append(self._running_log["ref_loss"].item())
            self.log[phase]["lang_loss"].append(self._running_log["lang_loss"].item())
            self.log[phase]["objectness_loss"].append(self._running_log["objectness_loss"].item())
            self.log[phase]["vote_loss"].append(self._running_log["vote_loss"].item())
            self.log[phase]["box_loss"].append(self._running_log["box_loss"].item())

            self.log[phase]["lang_acc"].append(self._running_log["lang_acc"])
            self.log[phase]["ref_acc"].append(self._running_log["ref_acc"])
            self.log[phase]["obj_acc"].append(self._running_log["obj_acc"])
            self.log[phase]["pos_ratio"].append(self._running_log["pos_ratio"])
            self.log[phase]["neg_ratio"].append(self._running_log["neg_ratio"])
            self.log[phase]["iou_rate_0.25"].append(self._running_log["iou_rate_0.25"])
            self.log[phase]["iou_rate_0.5"].append(self._running_log["iou_rate_0.5"])                

            # report
            if phase == "train":
                iter_time = self.log[phase]["fetch"][-1]
                iter_time += self.log[phase]["forward"][-1]
                iter_time += self.log[phase]["backward"][-1]
                iter_time += self.log[phase]["eval"][-1]
                self.log[phase]["iter_time"].append(iter_time)
                if (self._global_iter_id + 1) % self.verbose == 0:
                    self._train_report(epoch_id)

                # evaluation
                if self._global_iter_id != 0 and  self._global_iter_id % self.val_step == 0:
                    print("evaluating...")
                    # val
                    self._feed(self.dataloader["val"], "val", epoch_id)
                    self._dump_log("val")
                    self._set_phase("train")
                    self._epoch_report(epoch_id)

                # dump log
                self._dump_log("train")
                self._global_iter_id += 1


        # check best
        if phase == "val":
            cur_criterion = "iou_rate_0.5"
            cur_best = np.mean(self.log[phase][cur_criterion])
            if cur_best > self.best[cur_criterion]:
                self._log("best {} achieved: {}".format(cur_criterion, cur_best))
                self._log("current train_loss: {}".format(np.mean(self.log["train"]["loss"])))
                self._log("current val_loss: {}".format(np.mean(self.log["val"]["loss"])))
                self.best["epoch"] = epoch_id + 1
                self.best["loss"] = np.mean(self.log[phase]["loss"])
                self.best["ref_loss"] = np.mean(self.log[phase]["ref_loss"])
                self.best["lang_loss"] = np.mean(self.log[phase]["lang_loss"])
                self.best["objectness_loss"] = np.mean(self.log[phase]["objectness_loss"])
                self.best["vote_loss"] = np.mean(self.log[phase]["vote_loss"])
                self.best["box_loss"] = np.mean(self.log[phase]["box_loss"])
                self.best["lang_acc"] = np.mean(self.log[phase]["lang_acc"])
                self.best["ref_acc"] = np.mean(self.log[phase]["ref_acc"])
                self.best["obj_acc"] = np.mean(self.log[phase]["obj_acc"])
                self.best["pos_ratio"] = np.mean(self.log[phase]["pos_ratio"])
                self.best["neg_ratio"] = np.mean(self.log[phase]["neg_ratio"])
                self.best["iou_rate_0.25"] = np.mean(self.log[phase]["iou_rate_0.25"])
                self.best["iou_rate_0.5"] = np.mean(self.log[phase]["iou_rate_0.5"])

                # save model
                self._log("saving best models...\n")
                model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
                torch.save(self.model.state_dict(), os.path.join(model_root, "model.pth"))

    def _eval(self, data_dict):
        # dump
        self._running_log["lang_acc"] = data_dict["lang_acc"].item()
        self._running_log["ref_acc"] = np.mean(data_dict["ref_acc"])
        self._running_log["obj_acc"] = data_dict["obj_acc"].item()
        self._running_log["pos_ratio"] = data_dict["pos_ratio"].item()
        self._running_log["neg_ratio"] = data_dict["neg_ratio"].item()
        self._running_log["iou_rate_0.25"] = np.mean(data_dict["ref_iou_rate_0.25"])
        self._running_log["iou_rate_0.5"] = np.mean(data_dict["ref_iou_rate_0.5"])

    def _dump_log(self, phase):
        log = {
            "loss": ["loss", "ref_loss", "lang_loss", "objectness_loss", "vote_loss", "box_loss"],
            "score": ["lang_acc", "ref_acc", "obj_acc", "pos_ratio", "neg_ratio", "iou_rate_0.25", "iou_rate_0.5"]
        }
        for key in log:
            for item in log[key]:
                self._log_writer[phase].add_scalar(
                    "{}/{}".format(key, item),
                    np.mean([v for v in self.log[phase][item]]),
                    self._global_iter_id
                )

    def _finish(self, epoch_id):
        # print best
        self._best_report()

        # save model
        self._log("saving last models...\n")
        model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
        torch.save(self.model.state_dict(), os.path.join(model_root, "model_last.pth"))

        # export
        for phase in ["train", "val"]:
            self._log_writer[phase].export_scalars_to_json(os.path.join(CONF.PATH.OUTPUT, self.stamp, "tensorboard/{}".format(phase), "all_scalars.json"))

    def _train_report(self, epoch_id):
        # compute ETA
        fetch_time = self.log["train"]["fetch"]
        forward_time = self.log["train"]["forward"]
        backward_time = self.log["train"]["backward"]
        eval_time = self.log["train"]["eval"]
        iter_time = self.log["train"]["iter_time"]

        mean_train_time = np.mean(iter_time)
        mean_est_val_time = np.mean([fetch + forward for fetch, forward in zip(fetch_time, forward_time)])
        eta_sec = (self._total_iter["train"] - self._global_iter_id - 1) * mean_train_time
        eta_sec += len(self.dataloader["val"]) * np.ceil(self._total_iter["train"] / self.val_step) * mean_est_val_time
        eta = decode_eta(eta_sec)

        # print report
        iter_report = self.__iter_report_template.format(
            epoch_id=epoch_id + 1,
            iter_id=self._global_iter_id + 1,
            total_iter=self._total_iter["train"],
            train_loss=round(np.mean([v for v in self.log["train"]["loss"]]), 5),
            train_ref_loss=round(np.mean([v for v in self.log["train"]["ref_loss"]]), 5),
            train_lang_loss=round(np.mean([v for v in self.log["train"]["lang_loss"]]), 5),
            train_objectness_loss=round(np.mean([v for v in self.log["train"]["objectness_loss"]]), 5),
            train_vote_loss=round(np.mean([v for v in self.log["train"]["vote_loss"]]), 5),
            train_box_loss=round(np.mean([v for v in self.log["train"]["box_loss"]]), 5),
            train_lang_acc=round(np.mean([v for v in self.log["train"]["lang_acc"]]), 5),
            train_ref_acc=round(np.mean([v for v in self.log["train"]["ref_acc"]]), 5),
            train_obj_acc=round(np.mean([v for v in self.log["train"]["obj_acc"]]), 5),
            train_pos_ratio=round(np.mean([v for v in self.log["train"]["pos_ratio"]]), 5),
            train_neg_ratio=round(np.mean([v for v in self.log["train"]["neg_ratio"]]), 5),
            train_iou_rate_25=round(np.mean([v for v in self.log["train"]["iou_rate_0.25"]]), 5),
            train_iou_rate_5=round(np.mean([v for v in self.log["train"]["iou_rate_0.5"]]), 5),
            mean_fetch_time=round(np.mean(fetch_time), 5),
            mean_forward_time=round(np.mean(forward_time), 5),
            mean_backward_time=round(np.mean(backward_time), 5),
            mean_eval_time=round(np.mean(eval_time), 5),
            mean_iter_time=round(np.mean(iter_time), 5),
            eta_h=eta["h"],
            eta_m=eta["m"],
            eta_s=eta["s"]
        )
        self._log(iter_report)

    def _epoch_report(self, epoch_id):
        self._log("epoch [{}/{}] done...".format(epoch_id+1, self.epoch))
        epoch_report = self.__epoch_report_template.format(
            train_loss=round(np.mean([v for v in self.log["train"]["loss"]]), 5),
            train_ref_loss=round(np.mean([v for v in self.log["train"]["ref_loss"]]), 5),
            train_lang_loss=round(np.mean([v for v in self.log["train"]["lang_loss"]]), 5),
            train_objectness_loss=round(np.mean([v for v in self.log["train"]["objectness_loss"]]), 5),
            train_vote_loss=round(np.mean([v for v in self.log["train"]["vote_loss"]]), 5),
            train_box_loss=round(np.mean([v for v in self.log["train"]["box_loss"]]), 5),
            train_lang_acc=round(np.mean([v for v in self.log["train"]["lang_acc"]]), 5),
            train_ref_acc=round(np.mean([v for v in self.log["train"]["ref_acc"]]), 5),
            train_obj_acc=round(np.mean([v for v in self.log["train"]["obj_acc"]]), 5),
            train_pos_ratio=round(np.mean([v for v in self.log["train"]["pos_ratio"]]), 5),
            train_neg_ratio=round(np.mean([v for v in self.log["train"]["neg_ratio"]]), 5),
            train_iou_rate_25=round(np.mean([v for v in self.log["train"]["iou_rate_0.25"]]), 5),
            train_iou_rate_5=round(np.mean([v for v in self.log["train"]["iou_rate_0.5"]]), 5),
            val_loss=round(np.mean([v for v in self.log["val"]["loss"]]), 5),
            val_ref_loss=round(np.mean([v for v in self.log["val"]["ref_loss"]]), 5),
            val_lang_loss=round(np.mean([v for v in self.log["val"]["lang_loss"]]), 5),
            val_objectness_loss=round(np.mean([v for v in self.log["val"]["objectness_loss"]]), 5),
            val_vote_loss=round(np.mean([v for v in self.log["val"]["vote_loss"]]), 5),
            val_box_loss=round(np.mean([v for v in self.log["val"]["box_loss"]]), 5),
            val_lang_acc=round(np.mean([v for v in self.log["val"]["lang_acc"]]), 5),
            val_ref_acc=round(np.mean([v for v in self.log["val"]["ref_acc"]]), 5),
            val_obj_acc=round(np.mean([v for v in self.log["val"]["obj_acc"]]), 5),
            val_pos_ratio=round(np.mean([v for v in self.log["val"]["pos_ratio"]]), 5),
            val_neg_ratio=round(np.mean([v for v in self.log["val"]["neg_ratio"]]), 5),
            val_iou_rate_25=round(np.mean([v for v in self.log["val"]["iou_rate_0.25"]]), 5),
            val_iou_rate_5=round(np.mean([v for v in self.log["val"]["iou_rate_0.5"]]), 5),
        )
        self._log(epoch_report)
    
    def _best_report(self):
        self._log("training completed...")
        best_report = self.__best_report_template.format(
            epoch=self.best["epoch"],
            loss=round(self.best["loss"], 5),
            ref_loss=round(self.best["ref_loss"], 5),
            lang_loss=round(self.best["lang_loss"], 5),
            objectness_loss=round(self.best["objectness_loss"], 5),
            vote_loss=round(self.best["vote_loss"], 5),
            box_loss=round(self.best["box_loss"], 5),
            lang_acc=round(self.best["lang_acc"], 5),
            ref_acc=round(self.best["ref_acc"], 5),
            obj_acc=round(self.best["obj_acc"], 5),
            pos_ratio=round(self.best["pos_ratio"], 5),
            neg_ratio=round(self.best["neg_ratio"], 5),
            iou_rate_25=round(self.best["iou_rate_0.25"], 5),
            iou_rate_5=round(self.best["iou_rate_0.5"], 5),
        )
        self._log(best_report)
        with open(os.path.join(CONF.PATH.OUTPUT, self.stamp, "best.txt"), "w") as f:
            f.write(best_report)
