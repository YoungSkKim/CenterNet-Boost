# Getting Started

This document provides tutorials to train and evaluate CenterNet-Boost. Before getting started, make sure you have finished [installation](INSTALL.md) and [dataset setup](DATA.md).
## Benchmark evaluation

First, download the models you want to evaluate from our [model zoo](MODEL_ZOO.md) and put them in `CenterNet-Boost_ROOT/models/`.

### KITTI

To evaluate the kitti dataset, first compile the evaluation tool (from [here](https://github.com/prclibo/kitti_eval)):

~~~
cd CenterNet-Boost_ROOT/src/tools/kitti_eval
g++ -o evaluate_object_3d_offline evaluate_object_3d_offline.cpp -O3
~~~

Then run the evaluation with pretrained model:

~~~
python test.py --exp_id 3dop --dataset kitti --kitti_split 3dop --load_model ../models/ddd_3dop_boost.pth
~~~

to evaluate the 3DOP split. For the subcnn split, change `--kitti_split` to `subcnn` and load the corresponding models.
Note that test time augmentation is not trivially applicable for 3D orientation.

## Training
We have packed all the training scripts in the [experiments](../experiments) folder.
The experiment names are correspond to the model name in the [model zoo](MODEL_ZOO.md).
The number of GPUs for each experiments can be found in the scripts and the model zoo.
In the case that you don't have 8 GPUs, you can follow the [linear learning rate rule](https://arxiv.org/abs/1706.02677) to scale the learning rate as batch size.

~~~
python main.py --exp_id kitti --batch_size 32 --master_batch 15 --lr 1.25e-4  --gpus 0,1
~~~

The default learning rate is `1.25e-4` for batch size `32` (on 4 GPUs).
By default, pytorch evenly splits the total batch size to each GPUs.
`--master_batch` allows using different batchsize for the master GPU, which usually costs more memory than other GPUs.
If it encounters GPU memory out, using slightly less batch size (e.g., `112` of `128`) with the same learning is fine.

If the training is terminated before finishing, you can use the same commond with `--resume` to resume training. It will found the lastest model with the same `exp_id`.