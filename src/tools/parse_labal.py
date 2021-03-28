import numpy as np
import os
import shutil

# save test label from pvrcnn
gt_ann_dir = '/home/user/data/YoungseokKim/Code/CenterNet-Boost/data/kitti/testing/pv_rcnn_test/'
save_ann_dir = '/home/user/data/YoungseokKim/Code/CenterNet-Boost/data/kitti/testing/label_test_0.99/'
for i in range(7518):
    ann_path = gt_ann_dir + '%06d.txt'%(i)
    anns = open(ann_path, 'r')
    save_path = save_ann_dir + '%06d.txt'%(i)

    save_tmp = []
    for ann_ind, txt in enumerate(anns):
        tmp = txt[:-1].split(' ')
        if float(tmp[15]) >= 0.99:
            save_tmp.append(tmp[:15])
    with open(save_path, 'a+'):
        np.savetxt(save_path, save_tmp, fmt='%s')

# save train label
# gt_ann_dir = '/home/ubuntu/data/YoungseokKim/Code/CenterNet-Boost/data/kitti/training/label_2/'
# save_dir = '/home/ubuntu/data/YoungseokKim/Code/CenterNet-Boost/data/kitti/training/label_train/'
# check_dir = '/home/ubuntu/data/YoungseokKim/Code/CenterNet-Boost/data/kitti/training/label_val/'
# for i in range(7518):
#     ann_path = gt_ann_dir + '%06d.txt'%(i)
#     check_path = check_dir + '%06d.txt'%(i)
#     save_path = save_dir + '%06d.txt'%(i)
#     if not os.path.exists(check_path):
#         shutil.copy(ann_path, save_path)

