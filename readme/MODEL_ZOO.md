# MODEL ZOO

### Common settings and notes

- The experiments are run with PyTorch 1.7.1, CUDA 11.0.
- Training and testing times are measured on our servers with 4 RTX 3090 GPUs (24 GB Memory).

### KITTI 3DOP split

|Model       |GPUs|Train time|Test time|2D-E|2D-M|2D-H|3D-E|3D-M|3D-H|BEV-E|BEV-M|BEV-H| Download |
|------------|----|----------|---------|----|----|----|-----|-----|-----|-----|-----|-----|----------|
|[ddd_3dop_boost](../experiments/KITTI_3Ddetection_boost.sh)|4   | 2h       |  42ms   |96.3|93.1|85.7|19.1 |13.3 |11.9 |28.0 |19.7 |17.3 | [model](https://drive.google.com/file/d/11BuB6LMfl3IaOLNiJzZ-nNsf92h7YPdN/view?usp=sharing)|

### nuScenes

| Model                    | GPUs |Train time| Test time | Val mAP |  NDS  |  Download | 
|--------------------------|------|----------|-----------|---------|-------|------------|
| [nuScenes_boost](../experiments/nuScenes_3Ddetection_boost.sh)| 4    |     -  |    28ms   | 31.63  | 34.74  | [model](https://drive.google.com/file/d/1-DlkIecN-R1VXj3l0_vxclPf-H7sfYnJ/view?usp=sharing) |

