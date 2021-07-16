#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# Developed by: Xieyuanli Chen
# Brief: clean the LiDAR scans using our LiDAR-based moving object segmentation method

import os
import sys
import yaml
import numpy as np
from tqdm import tqdm

from utils import load_vertex, load_labels, load_files

if __name__ == '__main__':
  # load config file
  config_filename = 'config/post-processing.yaml'
  if len(sys.argv) > 1:
    config_filename = sys.argv[1]
  
  if yaml.__version__ >= '5.1':
    config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
  else:
    config = yaml.load(open(config_filename))
  
  # raw clean folder root
  scan_root = config['scan_root']
  
  # specify moving object segmentation results folder
  mos_pred_root = config['mos_pred_root']
  
  # create a new folder for combined results
  clean_scan_root = config['clean_scan_root']
  
  # specify the split
  split = config['split']
  data_yaml = yaml.load(open('config/semantic-kitti-mos.yaml'))
  
  # create output folder
  seqs = []
  if not os.path.exists(os.path.join(clean_scan_root, "sequences")):
    os.makedirs(os.path.join(clean_scan_root, "sequences"))
  
  if split == 'train':
    for seq in data_yaml["split"]["train"]:
      seq = '{0:02d}'.format(int(seq))
      print("train", seq)
      if not os.path.exists(os.path.join(clean_scan_root, "sequences", seq, "clean_scans")):
        os.makedirs(os.path.join(clean_scan_root, "sequences", seq, "clean_scans"))
      seqs.append(seq)
  if split == 'valid':
    for seq in data_yaml["split"]["valid"]:
      seq = '{0:02d}'.format(int(seq))
      print("train", seq)
      if not os.path.exists(os.path.join(clean_scan_root, "sequences", seq, "clean_scans")):
        os.makedirs(os.path.join(clean_scan_root, "sequences", seq, "clean_scans"))
      seqs.append(seq)
  if split == 'test':
    for seq in data_yaml["split"]["test"]:
      seq = '{0:02d}'.format(int(seq))
      print("train", seq)
      if not os.path.exists(os.path.join(clean_scan_root, "sequences", seq, "clean_scans")):
        os.makedirs(os.path.join(clean_scan_root, "sequences", seq, "clean_scans"))
      seqs.append(seq)
  
  for seq in seqs:
    # load moving object segmentation files
    mos_pred_seq_path = os.path.join(mos_pred_root, "sequences", seq, "predictions")
    mos_pred_files = load_files(mos_pred_seq_path)
    
    # load semantic segmentation files
    raw_scan_path = os.path.join(scan_root, "sequences", seq, "velodyne")
    raw_scan_files = load_files(raw_scan_path)
    
    print('processing seq:', seq)
    
    for frame_idx in tqdm(range(len(mos_pred_files))):
      mos_pred, _ = load_labels(mos_pred_files[frame_idx])  # mos_pred should be 9/251 for static/dynamic
      current_scan = load_vertex(raw_scan_files[frame_idx])

      clean_scan = current_scan[mos_pred < 250]
      np.array(clean_scan, dtype=np.float32).tofile(os.path.join(clean_scan_root, "sequences",
                                                                 seq, "clean_scans", str(frame_idx).zfill(6) + '.bin'))
