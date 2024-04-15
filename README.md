# Generative-AI-based-radio-map-estimation
The project is created for generative AI-based radio map estimation. This includes point-level estimation and cell-level estimation methods.

## Requirement
Pytorch 1.4+  Tensorflow 2.2.0 +

## Dataset
We support both simulated and measured datasets.

### Simulated dataset
please download at [RadioMapSeer](https://ieee-dataport.org/documents/dataset-pathloss-and-toa-radio-maps-localization-application) 
We support the usage of the latest of these (2024) 3D radio map datasets (**RadioMap3DSeer**).

Please refer to DatasetPaper as follow for details.
@article{DatasetPaper,
url = {https://arxiv.org/abs/2212.11777},
journal={arXiv preprint:2212.11777},
author = {Yapar, {\c{C}}a{\u{g}}kan and Levie, Ron and Kutyniok, Gitta and Caire, Giuseppe},
title = {Dataset of Pathloss and {ToA} Radio Maps With Localization Application},
publisher = {arXiv},
year = {2022}
}

### Measured dataset
please download at [RSRPSet_urban](https://ieee-dataport.org/documents/rsrpseturban-radio-map-dense-urban) 

Please refer to DatasetPaper as follow for details.
@ARTICLE{10227351,
  author={Zheng, Yi and Wang, Ji and Li, Xingwang and Li, Jiping and Liu, Shouyin},
  journal={IEEE Transactions on Cognitive Communications and Networking}, 
  title={Cell-Level RSRP Estimation With the Image-to-Image Wireless Propagation Model Based on Measured Data}, 
  year={2023},
  volume={9},
  number={6},
  pages={1412-1423},
  keywords={Estimation;Predictive models;Wireless communication;Data models;Computational modeling;Buildings;Training;Reference signal receiving power;wireless propagation model;conditional generative adversarial networks},
  doi={10.1109/TCCN.2023.3307945}}

## Point level Model
We only support the Point level Model on the measured datasets.
The list of models we support:
(**Empirical wireless propagation model:"**) Cost231-Hata/SPM
(**machine learning based model:"**) Polynomial regression/KNN/Random forest/SVM (test)/Gaussian process regression

How to use:
  point-level estimation\plot python draw_feature_v2.py
  (*Select the corresponding model test by changing the TYPE variable.*)

## Cell level Model
We support the following models on the both above measured and simulated datasets.

### RadioUNet
(** At simulated dataset: **) cell-level estimation\simulation dataset task\radiomap\src python train.py --exp_id NAME_YOUR_GIVEN --arch RadioUnet --dataset radiomap(or radiomap3d)

### RadioGAN
(** At simulated dataset: **) cell-level estimation\simulation dataset task\radiomap\src python train.py --exp_id NAME_YOUR_GIVEN --arch RadioGan --dataset radiomap(or radiomap3d)

Still test:
(** RadioCycle: **) cell-level estimation\simulation dataset task\radiomap\src python train.py --exp_id NAME_YOUR_GIVEN --arch RadioCycle --dataset radiomap

(** RadioTrans: **) cell-level estimation\simulation dataset task\radiomap\src python train.py --exp_id NAME_YOUR_GIVEN --arch RadioTrans --dataset radiomap
