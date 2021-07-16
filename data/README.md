This is a folder to contains a toy dataset used for LiDAR-MOS

Please use the recommended data structure as follows:

```bash
  data
    ├── sequences
    │   └── 08
    │       ├── calib.txt                       # calibration file provided by KITTI
    │       ├── poses.txt                       # ground truth poses file provided by KITTI
    │       ├── velodyne                        # velodyne 64 LiDAR scans provided by KITTI
    │       │   ├── 000000.bin
    │       │   ├── 000001.bin
    │       │   └── ...
    │       ├── clean_scans                     # clean scans after applying our MOS results as masks
    │       │   ├── 000000.bin
    │       │   ├── 000001.bin
    │       │   └── ...
    │       ├── labels                          # ground truth labels provided by SemantiKITTI
    │       │   ├── 000000.label
    │       │   ├── 000001.label
    │       │   └── ...
    │       └── residual_images_1               # the proposed residual images
    │           ├── 000000.npy
    │           ├── 000001.npy
    │           └── ...
    ├── predictions_salsanext_residual_1_valid  # MOS results using SalsaNext with 1 residual images
    │   └── sequences
    │       └── 08    
    │           └── predictions
    │               ├── 000000.label
    │               ├── 000001.label
    │               └── ...
    ├── predictions_salsanext_sem_valid         # SalsaNext semantic segmentation predictions
    │   └── sequences
    │       └── 08    
    │           └── predictions
    │               ├── 000000.label
    │               ├── 000001.label
    │               └── ...
    └── model_salsanext_residual_1              # MOS pretrained model using SalsaNext with 1 residual images
        ├── arch_cfg.yaml
        ├── data_cfg.yaml
        └── SalsaNext_valid_best                 
```

