import os
from datetime import datetime
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
'''
    Created by DogyoonLee
    https://github.com/dogyoonlee/pointcloud_visualizer/tree/main/pointcloud_feature_visualization_open3d
'''
feature_clip = 0.1


class feature_vis():
    def __init__(self):
        self.save_path = self.save_path_create()

    def save_path_create(self):
        now = datetime.now()
        save_time_str = str(now.year) + str('-') + str(
            now.month) + str('-') + str(now.day) + str('-') + str(
                now.hour) + str('-') + str(now.minute)
        feature_save_path = os.path.join(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                'feature_vis'), save_time_str)
        if not os.path.exists(feature_save_path):
            os.makedirs(feature_save_path)
        return feature_save_path

    def feature_vis_normalization(self, matrix):
        # import pdb; pdb.set_trace()
        matrix -= np.min(matrix, axis=1, keepdims=True)
        matrix /= np.max(matrix, axis=1, keepdims=True)
        matrix = np.exp(matrix) - 0.99999999
        # matrix = np.clip(matrix, 0, 1)
        return matrix

    def feature_active_compute(self, features, compute_type='square_sum'):
        if compute_type is 'square_sum':
            compute_feature = np.sum(features**2, axis=2, keepdims=True)
        else:
            # Not yet implemented. square_sum duplicated
            compute_feature = np.sum(features**2, axis=2, keepdims=True)

        normalized_feature = self.feature_vis_normalization(compute_feature)
        return normalized_feature

    def color_mapping(self, score, color_type='Reds'):
        color_r = cm.get_cmap(color_type)
        return color_r(score)[:3]

    def score_to_rgb(self, feature_score):
        B, N, _ = feature_score.shape
        rgb_score = np.zeros((B, N, 3))
        for i in range(B):
            for j in range(N):
                rgb_score[i][j][:] = self.color_mapping(
                    score=feature_score[i][j][0], color_type='OrRd')
        return rgb_score

    def feature_save(self, xyz, features, layer_name='layer_1'):
        '''
            input: 
                xyz: B x N x 3
                features: B x N x C
            output:
                saved_file: B x N x (3 + 3)
        '''
        xyz = np.array(xyz.cpu())
        features = np.array(features.detach().cpu())
        B, N, Coord = xyz.shape
        _, C, _ = features.shape
        features = np.transpose(features, (0, 2, 1))
        features_active = self.feature_active_compute(
            features, compute_type='square_sum')

        features_rgb = self.score_to_rgb(feature_score=features_active)

        feature_vis_save = np.concatenate((xyz, features_rgb), axis=2)
        filename_base = 'feature_' + layer_name
        np.save(os.path.join(self.save_path, filename_base), feature_vis_save)
