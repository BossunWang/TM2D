from os.path import join as pjoin

from common.skeleton import Skeleton
import numpy as np
import os
from common.quaternion import *
from utils.paramUtil import *

import torch
from tqdm import tqdm

from tools.pose2feature_converter_22j import Pose2Feature22J
from tools.pose2feature_converter_24j import Pose2Feature24J


class Pose2FeatureConverter(object):
    def __init__(self):
        self.pose2feature22j = Pose2Feature22J()
        self.pose2feature24j = Pose2Feature24J()
        self.mean_22j = np.load('./checkpoints/t2m/Comp_v6_KLD005/meta/mean.npy')
        self.std_22j = np.load('./checkpoints/t2m/Comp_v6_KLD005/meta/std.npy')
        # self.mean_24j = np.load('./checkpoints/aistppml3d/VQVAEV3_motion0919/meta/mean.npy')
        # self.std_24j = np.load('./checkpoints/aistppml3d/VQVAEV3_motion0919/meta/std.npy')
        self.mean_24j = np.load('./checkpoints/aistppml3d/VQVAEV3_aistppml3d_motion_1003_d3/meta/mean.npy')
        self.std_24j = np.load('./checkpoints/aistppml3d/VQVAEV3_aistppml3d_motion_1003_d3/meta/std.npy')

    def normed_f24j_to_normed_f22j(self, nf24j):
        """
        Used in the dataloader to convert the features normalized from 24J to 22J.
        changing the fps from 60 to 20.
        :return:
        """
        f24j = self.inv_transform(nf24j, self.std_24j, self.mean_24j)
        pose_seq_24j = self.pose2feature24j.feature2pose(f24j, 24)
        pose_seq_22j = pose_seq_24j[::3, :22]  # 60fps to 20fps
        f22j = self.pose2feature22j.pose2feature(pose_seq_22j)
        nf22j = self.normalization(f22j, self.std_22j, self.mean_22j)
        return nf22j

    def joint60fps24j_to_f22j(self, pose_seq_24j):
        pose_seq_22j = pose_seq_24j[::3, :22]  # 60fps to 20fps
        f22j = self.pose2feature22j.pose2feature(pose_seq_22j)
        nf22j = self.normalization(f22j, self.std_22j, self.mean_22j)
        return nf22j

    def inv_transform(self, data, std, mean):
        return data * std + mean

    def normalization(self, data, std, mean):
        return (data - mean) / std
