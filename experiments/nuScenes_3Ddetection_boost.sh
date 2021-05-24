cd src
# train 
python main.py --exp_id nuScenes_CenterNet-Boost --dataset nuscenes --batch_size 64 --gpus 0,1,2,3 --lr 2.5e-4 --num_epochs 140 --lr_step 90,120 --save_point 90,120,140
# test
python test.py --exp_id nuScenes_CenterNet-Boost --dataset nuscenes --resume
cd ..