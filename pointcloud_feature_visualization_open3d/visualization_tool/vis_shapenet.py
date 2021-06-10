import open3d as o3d
import numpy as np
import os
import argparse
'''
    {'Airplane': 0,   *** 
    'Bag': 1, 
    'Cap': 2, 
    'Car': 3, 
    'Chair': 4,        ****
    'Earphone': 5, 
    'Guitar': 6, 
    'Knife': 7, 
    'Lamp': 8,     ***
    'Laptop': 9,
    'Motorbike': 10, 
    'Mug': 11, 
    'Pistol': 12, 
    'Rocket': 13, 
    'Skateboard': 14, 
    'Table': 15}
'''


def shapenet(file='', object='Airplane', layer='e', layer_num=1, idx=0):
    '''
        Visualization shapenet
        @ https://github.com/dogyoonlee/pointcloud_visualizer/tree/main/pointcloud_feature_visualization_open3d
    '''
    if file is '':
        if layer is 'e':
            filename = '../../feature_vis/2021-3-3-12/' + object + '/feature_Object_' + str(
                idx) + '_' + object + '_Encode_Layer_' + str(layer_num) + '.npy'
        else:
            filename = '../../feature_vis/2021-3-3-12/' + object + '/feature_Object_' + str(
                idx) + '_' + object + '_Decode_Layer_' + str(layer_num) + '.npy'
    else:
        filename = file
    # input: B x N x (3 + 3) ; rgb value format : 0~1
    data = np.load(filename)
    data = data[0][:][:]

    xyzrgb = data[:, :6]
    # xyzrgb[:, 3:] /= 255
    np.savetxt("shapenet.txt", xyzrgb)
    pcd = o3d.io.read_point_cloud("shapenet.txt", format='xyzrgb')
    # o3d.visualization.
    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj', default='Airplane', type=str)
    parser.add_argument('--layer', default='e', type=str)
    parser.add_argument('--layer_num', default=1, type=int)
    parser.add_argument('--idx', default=0, type=int)
    parser.add_argument('--file', default='', type=str)
    args = parser.parse_args()
    shapenet(file=args.file,
             object=args.obj,
             layer=args.layer,
             layer_num=args.layer_num,
             idx=args.idx)
