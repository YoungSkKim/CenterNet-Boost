import torch

def compute_depth_metrics(gt, pred):
    """
    Compute depth metrics from predicted and ground-truth depth maps

    Parameters
    ----------
    config : CfgNode
        Metrics parameters
    gt : torch.Tensor [B,1,H,W]
        Ground-truth depth map
    pred : torch.Tensor [B,1,H,W]
        Predicted depth map
    use_gt_scale : bool
        True if ground-truth median-scaling is to be used

    Returns
    -------
    metrics : torch.Tensor [7]
        Depth metrics (abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3)
    """

    # Initialize variables
    batch_size, _, gt_height, gt_width = gt.shape
    abs_diff = abs_rel = sq_rel = rmse = rmse_log = a0 = a1 = a2 = a3 = 0.0

    # Interpolate predicted depth to ground-truth resolution
    pred = torch.nn.functional.interpolate(pred, gt.shape[-2:], mode='bilinear', align_corners=True)

    # For each depth map
    for pred_i, gt_i in zip(pred, gt):
        gt_i, pred_i = torch.squeeze(gt_i), torch.squeeze(pred_i)
        # Keep valid pixels (min/max depth and crop)
        valid = (gt_i > 0) & (gt_i < 60)
        # valid = valid & crop_mask.bool() if crop else valid
        # Stop if there are no remaining valid pixels
        if valid.sum() == 0:
            continue
        # Keep only valid pixels
        gt_i, pred_i = gt_i[valid], pred_i[valid]
        # Clamp predicted depth values to min/max values
        pred_i = pred_i.clamp(0, 60)

        # Calculate depth metrics

        thresh = torch.max((gt_i / pred_i), (pred_i / gt_i))
        a0 += (thresh < 1.10     ).float().mean()
        a1 += (thresh < 1.25     ).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        diff_i = gt_i - pred_i
        abs_diff += torch.mean(torch.abs(diff_i))
        abs_rel += torch.mean(torch.abs(diff_i) / gt_i)
        sq_rel += torch.mean(diff_i ** 2 / gt_i)
        rmse += torch.sqrt(torch.mean(diff_i ** 2))
        rmse_log += torch.sqrt(torch.mean((torch.log(gt_i) -
                                           torch.log(pred_i)) ** 2))
    # Return average values for each metric
    return torch.tensor([metric / batch_size for metric in
        [abs_rel, sq_rel, rmse, rmse_log, a0, a1, a2, a3]]).type_as(gt)


def eval_depth(batch, output, metrics_all, metrics_obj):
    depth_gt_all = batch['auxdep'] * batch['auxdep_mask'][:, :, :, 0].unsqueeze(0)
    depth_gt_obj = batch['auxdep'] * batch['auxdep_mask'][:, :, :, 1].unsqueeze(0)

    # eval DORN
    # depth_dorn_path = '/home/user/data/Dataset/KITTI/training/dorn/%06d.png'%batch['meta']['img_id'][0].numpy()
    # depth_dorn = cv2.imread(depth_dorn_path, cv2.IMREAD_ANYDEPTH)
    # depth_dorn = (depth_dorn / 256.).astype(np.float32)
    # depth_dorn = torch.from_numpy(depth_dorn).unsqueeze(0).unsqueeze(0).to('cuda:0')
    # metrics_all.append(compute_depth_metrics(depth_gt_all, depth_dorn))
    # metrics_obj.append(compute_depth_metrics(depth_gt_obj, depth_dorn))

    # eval proposed method
    metrics_all.append(compute_depth_metrics(depth_gt_all, output['dep']))
    metrics_obj.append(compute_depth_metrics(depth_gt_obj, output['dep']))

    if len(metrics_all) == 3769:  # TODO: make len(dataset) as variable
        metrics_all = (sum(metrics_all) / len(metrics_all)).detach().cpu().numpy()
        metrics_obj = (sum(metrics_obj) / len(metrics_obj)).detach().cpu().numpy()
        names = ['abs_rel', 'sqr_rel', 'rmse', 'rmse_log', 'a0', 'a1', 'a2', 'a3']
        print('raw depth map')
        for name, metrics_all in zip(names, metrics_all):
            print('{} = {:.3f}'.format(name, metrics_all))
        print('object-centric depth map')
        for name, metrics_obj in zip(names, metrics_obj):
            print('{} = {:.3f}'.format(name, metrics_obj))
    return metrics_all, metrics_obj