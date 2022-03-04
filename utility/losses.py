import torch
import torch.nn as nn
import random

def l2_loss(pred_pose, pred_pose_gt, Mask_offset, Mask_pose, Mask_S, cfg, mode='average'):
    """
    Input:
        pred_pose: Tensor of shape (seq_len, batch, joints). Predicted trajectory.
        pred_pose_gt: Tensor of shape (seq_len, batch, joints). Groud truth
        Mask_offset, Mask_pose, Mask_S: masks of offset, pose and visibility features
        cfg: input params
        DB: dataset name
        mode: loss type
    Output:
        loss: l2 loss depending on mode
    """
    BCE_loss = False
    if cfg.dataset.dataset_name == "3dpw" or cfg.dataset.dataset_name == "3DPW":
        Mask = []   #3dpw does not have missing joints
    elif cfg.dataset.dataset_name == "posetrack":
        visib_loss = 0
        if cfg.trainmode.Add_visib == True and cfg.trainmode.Add_offset == True and cfg.trainmode.Add_pose == True:
            # test_pose_pred = convert2Pose(test_pred, lastseen_poses_val)   #***********
            if cfg.model.visib_loss == "BCE":
                BCE_loss = True
                Mask = torch.cat((Mask_offset, Mask_pose), 2)
            else:
                Mask = torch.cat((Mask_offset, Mask_pose, Mask_S), 2)
        elif cfg.trainmode.Add_visib == False and cfg.trainmode.Add_offset == True and cfg.trainmode.Add_pose == True:
            Mask = torch.cat((Mask_offset, Mask_pose), 2)
        elif cfg.trainmode.Add_visib == False and cfg.trainmode.Add_offset == False and cfg.trainmode.Add_pose == True:
            Mask = Mask_pose

    if not BCE_loss:
        # c = (pred_pose_gt.permute(1, 0, 2) - pred_pose.permute(1, 0, 2)) ** 2
        c = torch.sub(pred_pose_gt.permute(1, 0, 2), pred_pose.permute(1, 0, 2)) ** 2
    else:
        offset_pose_gt = pred_pose_gt.permute(1, 0, 2)[:, :, :int((pred_pose_gt.shape[2] / 5) * 4)]
        visib_gt = pred_pose_gt.permute(1, 0, 2)[:, :, int((pred_pose_gt.shape[2] / 5) * 4):]
        offset_pose_pred = pred_pose.permute(1, 0, 2)[:, :, :int((pred_pose.shape[2] / 5) * 4)]
        visib_pred = pred_pose.permute(1, 0, 2)[:, :, int((pred_pose.shape[2] / 5) * 4):]
        m = nn.Sigmoid()
        loss = nn.BCELoss()
        visib_loss = loss(m(visib_pred), visib_gt)
        c = torch.sub(offset_pose_gt, offset_pose_pred) ** 2


    if len(Mask) > 0:  #posetrack
        loss = torch.mul(c, Mask)
        if mode == 'sum':
            return torch.sum(loss)+visib_loss
        elif mode == 'average':
            return (torch.sum(loss) / torch.nonzero(Mask.data, as_tuple=False).size(0))+visib_loss
        elif mode == 'raw':
            return loss.sum(dim=2).sum(dim=1)+visib_loss
    else:
        loss = c
        if mode == 'sum':
            return torch.sum(loss)
        elif mode == 'average':
            return torch.mean(loss)
        elif mode == 'raw':
            return loss.sum(dim=2).sum(dim=1)

def get_total_norm(parameters, norm_type=2):
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            try:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm**norm_type
                total_norm = total_norm**(1. / norm_type)
            except:
                continue
    return total_norm