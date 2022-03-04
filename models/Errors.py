import numpy as np
from utility.utils import load_object

def computeMetrics_withpose(gts, cur_pred, cfg, Mask):
    """
    Function to compute metirc values with predicted poses
    Inputs:
        gts: array of shape (pred_len, model.jsize)
        cur_pred: array of shape (pred_len, model.jsize)
        cfg: input arguments
        Mask: visibility mask of pose, array of shape (pred_len, model.jsize)
    Output:
        errorPose_global:
    """

    gt_i_global = np.copy(gts)

    if cfg.dataset.dataset_name == "posetrack":
        errorPose_global = np.power(gt_i_global - cur_pred, 2) * Mask
        #get sum on joints and remove the effect of missing joints by averaging on visible joints
        errorPose_global = np.divide(np.sum(errorPose_global, 1), np.sum(Mask,axis=1))
        where_are_NaNs = np.isnan(errorPose_global)
        errorPose_global[where_are_NaNs] = 0
        errorPose_global = np.sqrt(errorPose_global)
    else:
        errorPose_global = np.power(gt_i_global - cur_pred, 2)
        errorPose_global = np.sum(errorPose_global, 1)
        errorPose_global = np.sqrt(errorPose_global)


    return errorPose_global

def GetErrors_PoseBased_withmask(pred, gts, cfg, Mask_pose):
    """
    Function to compute metirc values with predicted poses
    Inputs:
        pred: array of shape (pred_len, model.jsize)
        gts: array of shape (pred_len, model.jsize)
        cfg: input arguments
        Mask_pose: visibility mask of pose, array of shape (pred_len, model.jsize)
    Output:
        mean_errorsPose_global:
    """

    cur_pred_pose = pred
    # cur_pred_pose = Convert2Pose(cur_pred, lastseen_poses_val)

    # Now compute the l2 error.
    mean_errorsPose_global = computeMetrics_withpose(gts, cur_pred_pose, cfg, Mask_pose)

    return mean_errorsPose_global


