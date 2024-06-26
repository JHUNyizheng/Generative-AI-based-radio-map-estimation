# RadioMap Project

[comment]: <> (> [**Track to Detect and Segment: An Online Multi-Object Tracker**]&#40;http://arxiv.org/abs/2004.01177&#41;,            )
[**Track to Detect and Segment: An Online Multi-Object Tracker**](https://openaccess.thecvf.com/content/CVPR2021/papers/Wu_Track_To_Detect_and_Segment_An_Online_Multi-Object_Tracker_CVPR_2021_paper.pdf)  
Jialian Wu, Jiale Cao, Liangchen Song, Yu Wang, Ming Yang, Junsong Yuan        
In CVPR, 2021. [[Paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Wu_Track_To_Detect_and_Segment_An_Online_Multi-Object_Tracker_CVPR_2021_paper.pdf) [[Project Page]](https://jialianwu.com/projects/TraDeS.html) [Demo [(YouTube)](https://www.youtube.com/watch?v=oGNtSFHRZJAl) [(bilibili)](https://www.bilibili.com/video/BV12U4y1p7wg)]

<p align="left"> <img src='https://github.com/JialianW/homepage/blob/master/images/TraDeS_demo.gif?raw=true' align="center" width="400px">

## News

* As reported in the [OVIS](https://openreview.net/forum?id=IfzTefIU_3j) paper, TraDeS achieves competitive performance on Occluded Video Instance Segmentation (12.0 AP on OVIS test set).
* As reported in the [MvMHAT](https://www.researchgate.net/profile/Ruize-Han/publication/353819964_Self-supervised_Multi-view_Multi-Human_Association_and_Tracking/links/611356961ca20f6f8613727d/Self-supervised-Multi-view-Multi-Human-Association-and-Tracking.pdf) paper,
  TraDeS also performs well on Multi-view Persons Tracking.
* TraDeS has been applied to 6 datasets across 4 tasks through our or third-parties' implementations.

## Installation

Please refer to [INSTALL.md](readme/INSTALL.md) for installation instructions.

Please refer to [Data.md](readme/DATA.md) for dataset preparation.

## Run Demo
Before run the demo, first download our trained models:
[CrowdHuman model](https://drive.google.com/file/d/1pljgwSecg50OhCTc2yCEhEBY3AwvPFlp/view?usp=sharing) (2D tracking),
[MOT model](https://drive.google.com/file/d/18DQi6LqFuO7_2QObvZSNK2y_F8yXT17p/view?usp=sharing) (2D tracking) or [nuScenes model](https://drive.google.com/file/d/1PHcDPIvb6owVuMZKR_YieyYN12IhbQLl/view?usp=sharing) (3D tracking). 
Then, put the models in `TraDeS_ROOT/models/` and `cd TraDeS_ROOT/src/`. **The demo result will be saved as a video in `TraDeS_ROOT/results/`.**

## Evaluation and Training
**Training RadioUnet from simulation dataset**

    python train.py --dataset simulation --exp_id Interp_s5 --arch Interpolation --gpus 2

**Training RadioUnet from simulation dataset**

    python train.py --dataset simulation --exp_id Unet_s20 --arch RadioUnet --gpus 2

**Training RadioCycle from simulation dataset**

    python train.py --dataset simulation --exp_id cycleDWA_s5 --arch RadioCycle --gpus 2

**Training RadioYnet from simulation dataset**

    python train.py --dataset simulation --exp_id AEr_s5 --arch RadioYnet --gpus 2
    
**Training RadioYnet from Real dataset**

    python train.py --dataset real --exp_id AEr_s5 --arch RadioYnet --gpus 2

**Evaluation RadioYnet from simulation dataset**

    python test.py --dataset real --exp_id AEr_s5 --arch RadioYnet --gpus 2
### *2D Object Tracking*

| MOT17 Val                  | MOTA↑  |IDF1↑|IDS↓|
|-----------------------|----------|----------|----------|
| Our Baseline         |64.8|59.5|1055|
| [CenterTrack](https://arxiv.org/pdf/2004.01177.pdf)         |66.1|64.2|528|
| [TraDeS (ours)](experiments/mot17_test.sh)  |**68.2**|**71.7**|**285**|

**Test on MOT17 validation set:** Place the [MOT model](https://drive.google.com/file/d/18DQi6LqFuO7_2QObvZSNK2y_F8yXT17p/view?usp=sharing) in $TraDeS_ROOT/models/ and run:

    sh experiments/mot17_test.sh

**Train on MOT17 halftrain set:** Place the [pretrained model](https://drive.google.com/file/d/1pljgwSecg50OhCTc2yCEhEBY3AwvPFlp/view?usp=sharing) in $TraDeS_ROOT/models/ and run:

    sh experiments/mot17_train.sh

## *3D Object Tracking* 

| nuScenes Val                  | AMOTA↑|AMOTP↓|IDSA↓|
|-----------------------|----------|----------|----------|
| Our Baseline         |4.3|1.65|1792|
| [CenterTrack](https://arxiv.org/pdf/2004.01177.pdf)         |6.8|1.54|813|
| [TraDeS (ours)](experiments/nuScenes_test.sh) |**11.8**|**1.48**|**699**|

**Test on nuScenes validation set:** Place the [nuScenes model](https://drive.google.com/file/d/1PHcDPIvb6owVuMZKR_YieyYN12IhbQLl/view?usp=sharing) in $TraDeS_ROOT/models/. You need to change the MOT and nuScenes dataset API versions due to their conflicts. The default installed versions are for MOT dataset.  For experiments on nuScenes dataset, please run:

    sh nuscenes_switch_version.sh

    sh experiments/nuScenes_test.sh

To switch back to the API versions for MOT experiments, you can run:

    sh mot_switch_version.sh

**Train on nuScenes train set:** Place the [pretrained model](https://drive.google.com/file/d/1jGDrQ5I3ZxyGoep79egcT9MI3JM1ZKhG/view?usp=sharing) in $TraDeS_ROOT/models/ and run:
    
    sh experiments/nuScenes_train.sh

## *Train on Static Images*
We follow [CenterTrack](https://arxiv.org/pdf/2004.01177.pdf) which uses CrowdHuman to pretrain 2D object tracking model. Only the training set is used.

    sh experiments/crowdhuman.sh

The trained model is available at [CrowdHuman model](https://drive.google.com/file/d/1pljgwSecg50OhCTc2yCEhEBY3AwvPFlp/view?usp=sharing).


## Citation
If you find it useful in your research, please consider citing our paper as follows:

    @inproceedings{Wu2021TraDeS,
    title={Track to Detect and Segment: An Online Multi-Object Tracker},
    author={Wu, Jialian and Cao, Jiale and Song, Liangchen and Wang, Yu and Yang, Ming and Yuan, Junsong},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2021}}

## Acknowledgment
Many thanks to [CenterTrack](https://github.com/xingyizhou/CenterTrack) authors for their great framework!
