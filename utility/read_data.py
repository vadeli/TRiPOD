import numpy as np

from utility.utils import Normalize_data, Normalize_pose, save_object, load_object
from utility import utils
from utility import data_utils_posetrack, data_utils_3DPW
from utility.classes import *


def read_all_data_3DPW(cfg, load):
    """Loads data for training/testing and normalizes it.
    :param cfg: input arguments
    :return: offsetData: dictionary of ('train', 'test', 'validation') offset information for all the splits
          poseData: dictionary of ('train', 'test', 'validation') pose locations for all the splits
          shapedata: feature shape (#joints x and y)
          fileNames: dictionary of ('train', 'test', 'validation') filename of each sequence used, for all the splits
          visibData: dictionary of ('train', 'test', 'validation') visibility feature (S)
          dummy: int dummy value
          miss_masks: class structure of missinf mask for pose and offset for all the splits
          scaler_offset: offset Scaler containing statistical information for normalization
          scaler_pose: pose Scaler containing statistical information for normalization
    """
    if load == 0:
        # initialize data structure
        offsetData = {
            'train': [],
            'test': [],
            'validation': []
        }
        poseData = {
            'train': [],
            'test': [],
            'validation': []
        }
        fileNames = {
            'train': [],
            'test': [],
            'validation': []
        }


        # === Read training data ===
        print("Reading training data")
        poseData['train'], shapedata, fileNames['train'], complete_train_pose = data_utils_3DPW.load_data('train', cfg)
        print("Reading testing data")
        poseData['test'], shapedata, fileNames['test'], complete_test_pose = data_utils_3DPW.load_data('test', cfg)
        print("Reading validating data")
        poseData['validation'], shapedata, fileNames['validation'], complete_validation_pose  = data_utils_3DPW.load_data('validation', cfg)
        print("Preprocessing data")

        # Compute normalization stats
        poseData, scaler_pose = Normalize_pose(cfg, poseData, complete_train_pose, complete_test_pose,
                                                                          complete_validation_pose)

        if cfg.trainmode.Add_offset:
            offsetData = utils.GetVelocity(poseData, offsetData)


        print("done reading data.")
        save_object(offsetData, './data/3DPW/offsetData_'+ cfg.experiment.normalization + '.pkl')
        save_object(poseData, './data/3DPW/poseData_' + cfg.experiment.normalization + '.pkl')
        save_object(fileNames, './data/3DPW/fileNames.pkl')
        save_object(scaler_pose, './data/3DPW/scaler_pose_' + cfg.experiment.normalization + '.pkl')
    else:
        offsetData = load_object('./data/3DPW/offsetData_' + cfg.experiment.normalization + '.pkl')
        poseData = load_object('./data/3DPW/poseData_' + cfg.experiment.normalization + '.pkl')
        fileNames = load_object('./data/3DPW/fileNames.pkl')
        scaler_pose = load_object('./data/3DPW/scaler_pose_' + cfg.experiment.normalization + '.pkl')
        shapedata = offsetData['train'][0].shape[2]
        print("done reading data.")

    return offsetData, poseData, shapedata, fileNames, scaler_pose