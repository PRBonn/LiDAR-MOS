#!/usr/bin/env python3
# Developed by Xieyuanli Chen
# This file is covered by the LICENSE file in the root of this project.
# Brief: This script generates residual images

import os
import sys
import yaml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import load_poses, load_calib, load_files, load_vertex

try:
  from c_gen_virtual_scan import gen_virtual_scan as range_projection
except:
  print("Using clib by $export PYTHONPATH=$PYTHONPATH:<path-to-library>")
  print("Currently using python-lib to generate range images.")
  from utils import range_projection


if __name__ == '__main__':
  # load config file
  config_filename = 'config/data_preparing.yaml'
  if len(sys.argv) > 1:
    config_filename = sys.argv[1]
  
  if yaml.__version__ >= '5.1':
    config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
  else:
    config = yaml.load(open(config_filename))
  
  # specify parameters
  num_frames = config['num_frames']
  debug = config['debug']
  normalize = config['normalize']
  num_last_n = config['num_last_n']
  visualize = config['visualize']
  visualization_folder = config['visualization_folder']
  
  # specify the output folders
  residual_image_folder = config['residual_image_folder']
  if not os.path.exists(residual_image_folder):
    os.makedirs(residual_image_folder)
    
  if visualize:
    if not os.path.exists(visualization_folder):
      os.makedirs(visualization_folder)
  
  # load poses
  pose_file = config['pose_file']
  poses = np.array(load_poses(pose_file))
  inv_frame0 = np.linalg.inv(poses[0])
  
  # load calibrations
  calib_file = config['calib_file']
  T_cam_velo = load_calib(calib_file)
  T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
  T_velo_cam = np.linalg.inv(T_cam_velo)
  
  # convert kitti poses from camera coord to LiDAR coord
  new_poses = []
  for pose in poses:
    new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))
  poses = np.array(new_poses)
  
  # load LiDAR scans
  scan_folder = config['scan_folder']
  scan_paths = load_files(scan_folder)
  
  # test for the first N scans
  if num_frames >= len(poses) or num_frames <= 0:
    print('generate training data for all frames with number of: ', len(poses))
  else:
    poses = poses[:num_frames]
    scan_paths = scan_paths[:num_frames]
  
  range_image_params = config['range_image']
  
  # generate residual images for the whole sequence
  for frame_idx in tqdm(range(len(scan_paths))):
    file_name = os.path.join(residual_image_folder, str(frame_idx).zfill(6))
    diff_image = np.full((range_image_params['height'], range_image_params['width']), 0,
                             dtype=np.float32)  # [H,W] range (0 is no data)
    
    # for the first N frame we generate a dummy file
    if frame_idx < num_last_n:
      np.save(file_name, diff_image)
    
    else:
      # load current scan and generate current range image
      current_pose = poses[frame_idx]
      current_scan = load_vertex(scan_paths[frame_idx])
      current_range = range_projection(current_scan.astype(np.float32),
                                       range_image_params['height'], range_image_params['width'],
                                       range_image_params['fov_up'], range_image_params['fov_down'],
                                       range_image_params['max_range'], range_image_params['min_range'])[:, :, 3]
      
      # load last scan, transform into the current coord and generate a transformed last range image
      last_pose = poses[frame_idx - num_last_n]
      last_scan = load_vertex(scan_paths[frame_idx - num_last_n])
      last_scan_transformed = np.linalg.inv(current_pose).dot(last_pose).dot(last_scan.T).T
      last_range_transformed = range_projection(last_scan_transformed.astype(np.float32),
                                                range_image_params['height'], range_image_params['width'],
                                                range_image_params['fov_up'], range_image_params['fov_down'],
                                                range_image_params['max_range'], range_image_params['min_range'])[:, :, 3]
      
      # generate residual image
      valid_mask = (current_range > range_image_params['min_range']) & \
                   (current_range < range_image_params['max_range']) & \
                   (last_range_transformed > range_image_params['min_range']) & \
                   (last_range_transformed < range_image_params['max_range'])
      difference = np.abs(current_range[valid_mask] - last_range_transformed[valid_mask])
      
      if normalize:
        difference = np.abs(current_range[valid_mask] - last_range_transformed[valid_mask]) / current_range[valid_mask]

      diff_image[valid_mask] = difference
      
      if debug:
        fig, axs = plt.subplots(3)
        axs[0].imshow(last_range_transformed)
        axs[1].imshow(current_range)
        axs[2].imshow(diff_image, vmin=0, vmax=10)
        plt.show()
        
      if visualize:
        fig = plt.figure(frameon=False, figsize=(16, 10))
        fig.set_size_inches(20.48, 0.64)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(diff_image, vmin=0, vmax=1)
        image_name = os.path.join(visualization_folder, str(frame_idx).zfill(6))
        plt.savefig(image_name)
        plt.close()

      # save residual image
      np.save(file_name, diff_image)
