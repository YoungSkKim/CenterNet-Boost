# Dataset preparation

If you want to reproduce the results in the paper for benchmark evaluation or training, you will need to setup datasets.

### KITTI

- Download [images](http://www.cvlibs.net/download.php?file=data_object_image_2.zip), [annotations](http://www.cvlibs.net/download.php?file=data_object_label_2.zip), and [calibrations](http://www.cvlibs.net/download.php?file=data_object_calib.zip) from [KITTI website](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and unzip.

- Download the train-val split of [3DOP](https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz) and [SubCNN](https://github.com/tanshen/SubCNN/tree/master/fast-rcnn/data/KITTI) and place the data as below

- To provide ease of use, we provide [depth annotation]((https://drive.google.com/file/d/1Y6ps20Tl_BaFT_rCTYHBvStkrN0cmuQ9/view?usp=sharing)) of KITTI training set

  ~~~
  ${CenterNet-Boost_ROOT}
  |-- data
  `-- |-- kitti
      `-- |-- training
          |   |-- image_2
          |   |-- label_2
          |   |-- calib
          |   |-- depth_gt
          |-- ImageSets_3dop
          |   |-- test.txt
          |   |-- train.txt
          |   |-- val.txt
          |   |-- trainval.txt
          `-- ImageSets_subcnn
              |-- test.txt
              |-- train.txt
              |-- val.txt
              |-- trainval.txt
  ~~~

- Run `python convert_kitti_to_coco.py` in `tools` to convert the annotation into COCO format. You can set `DEBUG=True` in `line 5` to visualize the annotation.

- Link image folder

  ~~~
  cd ${CenterNet-Boost_ROOT}/data/kitti/
  mkdir images
  ln -s training/image_2 images/trainval
  ~~~

- The data structure should look like:

  ~~~
  ${CenterNet-Boost_ROOT}
  |-- data
  `-- |-- kitti
      `-- |-- annotations
          |   |-- kitti_3dop_train.json
          |   |-- kitti_3dop_val.json
          |   |-- kitti_subcnn_train.json
          |   |-- kitti_subcnn_val.json
          `-- images
              |-- trainval
              |-- test
  ~~~


## References
Please cite the corresponding References if you use the datasets.

~~~
  @INPROCEEDINGS{Geiger2012CVPR,
    author = {Andreas Geiger and Philip Lenz and Raquel Urtasun},
    title = {Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
    booktitle = {CVPR},
    year = {2012}
  }
~~~