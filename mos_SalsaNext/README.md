[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/salsanext-fast-semantic-segmentation-of-lidar/3d-semantic-segmentation-on-semantickitti)](https://paperswithcode.com/sota/3d-semantic-segmentation-on-semantickitti?p=salsanext-fast-semantic-segmentation-of-lidar) [![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2003.03653)

# SalsaNext: Fast, Uncertainty-aware Semantic Segmentation of LiDAR Point Clouds for Autonomous Driving

## Abstract 

In this paper, we introduce SalsaNext for the uncertainty-aware semantic segmentation of a full 3D LiDAR point cloud in real-time. SalsaNext is the next version of SalsaNet which has an encoder-decoder architecture where the encoder unit has a set of ResNet blocks and the decoder part combines upsampled features from the residual blocks. In contrast to SalsaNet, we introduce a new context module, replace the ResNet encoder blocks with a new residual dilated convolution stack with gradually increasing receptive fields and add the pixel-shuffle layer in the decoder. Additionally, we switch from stride convolution to average pooling and also apply central dropout treatment. To directly optimize the Jaccard index, we further combine the weighted cross-entropy loss with Lovasz-Softmax loss . We finally inject a Bayesian treatment to compute the epistemic and aleatoric uncertainties for each point in the cloud. We provide a thorough quantitative evaluation on the Semantic-KITTI dataset, which demonstrates that the proposed SalsaNext outperforms other state-of-the-art semantic segmentation.
## Examples 
![Example Gif](/images/SalsaNext.gif)

### Video 
[![Inference of Sequence 13](https://img.youtube.com/vi/MlSaIcD9ItU/0.jpg)](http://www.youtube.com/watch?v=MlSaIcD9ItU)



### Semantic Kitti Segmentation Scores

The up-to-date scores can be found in the Semantic-Kitti [page](http://semantic-kitti.org/tasks.html#semseg).

## How to use the code

First create the anaconda env with:
```conda env create -f salsanext_cuda10.yml --name salsanext``` then activate the environment with ```conda activate salsanext```.

To train/eval you can use the following scripts:


 * [Training script](train.sh) (you might need to chmod +x the file)
   * We have the following options:
     * ```-d [String]``` : Path to the dataset
     * ```-a [String]```: Path to the Architecture configuration file 
     * ```-l [String]```: Path to the main log folder
     * ```-n [String]```: additional name for the experiment
     * ```-c [String]```: GPUs to use (default ```no gpu```)
     * ```-u [String]```: If you want to train an Uncertainty version of SalsaNext (default ```false```) [Experimental: tests done so with uncertainty far used pretrained SalsaNext with [Deep Uncertainty Estimation](https://github.com/uzh-rpg/deep_uncertainty_estimation)]
   * For example if you have the dataset at ``/dataset`` the architecture config file in ``/salsanext.yml``
   and you want to save your logs to ```/logs``` to train "salsanext" with 2 GPUs with id 3 and 4:
     * ```./train.sh -d /dataset -a /salsanext.yml -m salsanext -l /logs -c 3,4```
<br>
<br>

 * [Eval script](eval.sh) (you might need to chmod +x the file)
   * We have the following options:
     * ```-d [String]```: Path to the dataset
     * ```-p [String]```: Path to save label predictions
     * ``-m [String]``: Path to the location of saved model
     * ``-s [String]``: Eval on Validation or Train (standard eval on both separately)
     * ```-u [String]```: If you want to infer using an Uncertainty model (default ```false```)
     * ```-c [Int]```: Number of MC sampling to do (default ```30```)
   * If you want to infer&evaluate a model that you saved to ````/salsanext/logs/[the desired run]```` and you
   want to infer$eval only the validation and save the label prediction to ```/pred```:
     * ```./eval.sh -d /dataset -p /pred -m /salsanext/logs/[the desired run] -s validation -n salsanext```
     
### Pretrained Model

[SalsaNext](https://cutt.ly/bpadjGj)
     
### Disclamer

We based our code on [RangeNet++](https://github.com/PRBonn/lidar-bonnetal), please go show some support!
 

### Citation

```
@misc{cortinhal2020salsanext,
    title={SalsaNext: Fast, Uncertainty-aware Semantic Segmentation of LiDAR Point Clouds for Autonomous Driving},
    author={Tiago Cortinhal and George Tzelepis and Eren Erdal Aksoy},
    year={2020},
    eprint={2003.03653},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

