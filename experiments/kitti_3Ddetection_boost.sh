cd src
# train 
python main.py --exp_id kitti_CenterNet-Boost --load_model ../models/nuScenes_3Ddetection_boost.pth --dataset kitti --batch_size 32 --gpus 0,1,2,3 --lr 1.25e-4 --num_epochs 70 --lr_step 60 --save_point 60,70
# test
python test.py --exp_id kitti_CenterNet-Boost --dataset kitti --resume
cd ..