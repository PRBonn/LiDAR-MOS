#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# Developed by Xieyuanli Chen
# Brief: This script combines moving object segmentation with semantic information

import os
import sys
import yaml
import numpy as np
from tqdm import tqdm

from utils import load_files, load_labels


if __name__ == '__main__':
  # load config file
  config_filename = 'config/post-processing.yaml'
  if len(sys.argv) > 1:
    config_filename = sys.argv[1]
  
  if yaml.__version__ >= '5.1':
    config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
  else:
    config = yaml.load(open(config_filename))
  
  # specify moving object segmentation results folder
  mos_pred_root = config['mos_pred_root']
  
  # specify semantic segmentation results folder
  semantic_pred_root = config['semantic_pred_root']
  
  # create a new folder for combined results
  combined_results_root = config['combined_results_root']
  
  # specify the split
  split = config['split']
  data_yaml = yaml.load(open('config/semantic-kitti-mos.yaml'))
  
  # create output folder
  seqs = []
  if not os.path.exists(os.path.join(combined_results_root, "sequences")):
    os.makedirs(os.path.join(combined_results_root, "sequences"))
  
  if split == 'train':
    print(data_yaml["split"]["train"])
    for seq in data_yaml["split"]["train"]:
      seq = '{0:02d}'.format(int(seq))
      print("train", seq)
      if not os.path.exists(os.path.join(combined_results_root, "sequences", seq, "clean_scans")):
        os.makedirs(os.path.join(combined_results_root, "sequences", seq, "clean_scans"))
      seqs.append(seq)
  if split == 'valid':
    for seq in data_yaml["split"]["valid"]:
      seq = '{0:02d}'.format(int(seq))
      print("train", seq)
      if not os.path.exists(os.path.join(combined_results_root, "sequences", seq, "clean_scans")):
        os.makedirs(os.path.join(combined_results_root, "sequences", seq, "clean_scans"))
      seqs.append(seq)
  if split == 'test':
    for seq in data_yaml["split"]["test"]:
      seq = '{0:02d}'.format(int(seq))
      print("train", seq)
      if not os.path.exists(os.path.join(combined_results_root, "sequences", seq, "clean_scans")):
        os.makedirs(os.path.join(combined_results_root, "sequences", seq, "clean_scans"))
      seqs.append(seq)
  
  for seq in seqs:
    # load moving object segmentation files
    mos_pred_seq_path = os.path.join(mos_pred_root, "sequences", seq, "predictions")
    mos_pred_files = load_files(mos_pred_seq_path)
    
    # load semantic segmentation files
    semantic_pred_seq_path = os.path.join(semantic_pred_root, "sequences", seq, "predictions")
    semantic_pred_files = load_files(semantic_pred_seq_path)
    
    print('processing seq:', seq)
    
    for frame_idx in tqdm(range(len(mos_pred_files))):
      mos_pred, _ = load_labels(mos_pred_files[frame_idx])  # mos_pred should be 9/251 for static/dynamic
      semantic_pred, _ = load_labels(semantic_pred_files[frame_idx])  # mos_pred should be full classes
      semantic_pred_mapped = np.ones(len(mos_pred), dtype=np.uint32) * 9
      combine_pred = np.ones(len(mos_pred), dtype=np.uint32) * 9
      
      # mapping semantic into static and movable classes
      movable_mask = (semantic_pred > 0) & (semantic_pred < 40)
      semantic_pred_mapped[movable_mask] = 251
      
      # if consistent keep the same, otherwise labeled as static
      combined_mask = (semantic_pred_mapped == mos_pred)
      combine_pred[combined_mask] = mos_pred[combined_mask]
  
      file_name = os.path.join(combined_results_root, "sequences", seq, "predictions", str(frame_idx).zfill(6))
      combine_pred.reshape((-1)).astype(np.uint32)
      combine_pred.tofile(file_name + '.label')