# Installation


The code was tested on Ubuntu 16.04, with [Anaconda](https://www.anaconda.com/download) Python 3.8, CUDA 11.0, and [PyTorch]((http://pytorch.org/)) v1.7.1.
(you will need to switch DCNv2 version for PyTorch <1.0).
After installing Anaconda:

0. [Optional but highly recommended] create a new conda environment. 

    ~~~
    conda create --name CenterNet-Boost python=3.8
    ~~~
    And activate the environment.
    
    ~~~
    conda activate CenterNet-Boost
    ~~~

1. Install PyTorch:

    ~~~
    conda install pytorch torchvision -c pytorch
    ~~~
    

2. Install [COCOAPI](https://github.com/cocodataset/cocoapi):

    ~~~
    pip install cython; pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
    ~~~

3. Clone this repo:

    ~~~
    CenterTrack_ROOT=/path/to/clone/CenterNet-Boost
    git clone --recursive https://github.com/YoungSkKim/CenterNet-Boost $CenterNet-Boost_ROOT
    ~~~

    You can manually install the [submodules](../.gitmodules) if you forget `--recursive`.

4. Install the requirements

    ~~~
    pip install -r requirements.txt
    ~~~
    
    
5. Compile deformable convolutional (from [DCNv2](https://github.com/CharlesShang/DCNv2/)).

    ~~~
    cd $CenterTrack_ROOT/src/lib/model/networks/
    # git clone https://github.com/CharlesShang/DCNv2/ # clone if it is not automatically downloaded by `--recursive`.
    cd DCNv2
    ./make.sh
    ~~~

6. Download pertained models for [KITTI](https://drive.google.com/file/d/11BuB6LMfl3IaOLNiJzZ-nNsf92h7YPdN/view?usp=sharing) or [nuScenes](https://drive.google.com/file/d/1-DlkIecN-R1VXj3l0_vxclPf-H7sfYnJ/view?usp=sharing) and move them to `$CenterNet-Boost_ROOT/models/`. More models can be found in [Model zoo](MODEL_ZOO.md).