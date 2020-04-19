# Incremental Few Shot Object Detection
Implementation of the paper:

```
@inproceedings{perez2020incremental,
  title={Incremental Few-Shot Object Detection},
  author={Perez-Rua, Juan-Manuel and Zhu, Xiatian and Hospedales, Timothy and Xiang, Tao},
  booktitle={CVPR},
  year={2020}
}
```

This code builds on the [**Objects as Points**](http://arxiv.org/abs/1904.07850) paper and is forked from their official code repo.
Please cite both papers if you use this code.


## Abstract 

Most existing object detection methods rely on the availability of abundant labelled training samples per class and offline model training in a batch mode. These requirements substantially limit their scalability to open-ended accommodation of novel classes with limited labelled training data. We present a study aiming to go beyond these limitations by considering the Incremental Few-Shot Detection (iFSD) problem setting, where new classes must be registered incrementally (without revisiting base classes) and with few examples. To this end we propose OpeN-ended Centre nEt (ONCE), a detector designed for incrementally learning to detect novel class objects with few examples. This is achieved by an elegant adaptation of the CentreNet detector to the few-shot learning scenario, and meta-learning a class-specific code generator model for registering novel classes. ONCE fully respects the incremental learning paradigm, with novel class registration requiring only a single forward pass of few-shot training samples, and no access to base classes -- thus making it suitable for deployment on embedded devices. Extensive experiments conducted on both the standard object detection and fashion landmark detection tasks show the feasibility of iFSD for the first time, opening an interesting and very important line of research.


## Training ONCE

The following python scripts can be found in the `src` directory of this repo.

- Episodic training. The feature extractor and class code generator networks are both initialised from a 
model supervised on the COCO base classes. Each episode consiste on a 3-way (--n_class) detection problem with 5 shots (--k_shots).

~~~
python main_ep.py epdet --exp_id coco_ep_res50_ONCE --dataset coco_ep --arch resmetafull_50 --batch_size 32 --master_batch 4 --lr 1e-4 --gpus 0,1,2,3,4,5,6,7 --num_workers 24 --head_conv 0 --n_class 3 --k_shots 5 --cat_spec_wh --load_basemodel ../models/coco_res50_nohc_few/model_30.pth --load_metamodel ../models/coco_res50_nohc_few/model_30.pth --metasize 50
~~~

- Testing the trained ONCE model by running 100 experiments (20-way with 5-shots):

~~~
python test_ep.py epdet --exp_id coco_ep_res50_ONCE_test --dataset coco_ep --arch resmeta_50 --gpus 0,1,2,3,4,5,6,7 --num_workers 24 --head_conv 0 --k_shots 5  --load_model ../exp/epdet/coco_ep_res50_ONCE/model_best.pth --metasize 50 --cat_spec_wh --num_test_iters 100 --flip_test --keep_res --test_scales 0.75,1,1.25
~~~

- The 1 by 1 incremental setting can be tested from the same ONCE model by running:

~~~
python test_inc.py incdet --exp_id coco_ep_res50_ONCE_test_inc --dataset coco_ep --arch resmeta_50 --gpus 0,1,2,3,4,5,6,7 --num_workers 24 --head_conv 0 --k_shots 20  --load_model ../exp/epdet/coco_ep_res50_ONCE/model_best.pth --metasize 50 --cat_spec_wh --num_test_iters 1 --flip_test --keep_res --test_scales 0.75,1,1.25
~~~


