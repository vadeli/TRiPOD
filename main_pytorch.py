from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import sys
import argparse
import numpy as np


import torch

import logging

from yacs.config import CfgNode

from utility.utils import boolean


import train

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
sys.path.insert(1, os.getcwd())

print(torch.__version__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# summaries_dir = os.path.normpath(os.path.join(train_dir, "log"))  # Directory for TB summaries
# if FLAGS.load == 1:
#     FLAGS.ckpt_path ='./experiments/posetrack/out_14/depth_1/encoder_dim_256/lr_5e-06/non-context/social_gat_1_3_0.6/residual_vel/idx_34764/posetrack/'
#     summaries_dir = os.path.normpath(os.path.join(FLAGS.ckpt_path, "log"))
#     train_dir = FLAGS.ckpt_path
#     ckpt_path = os.path.join(FLAGS.ckpt_path,'best/14/')
# print(summaries_dir)


def main(cfg, summaries_dir):
    if cfg.experiment.sample:
        sample()
    else:
        if cfg.dataset.dataset_name == "posetrack":
            train.Curriculum_social_PT(cfg, summaries_dir)

        elif cfg.dataset.dataset_name == "3dpw" or cfg.dataset.dataset_name == "3DPW":
            rain.Curriculum_social_3DPW(cfg, summaries_dir)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument(
        "--cfg", dest="cfg_file", help="optional config file", type=str
    )
    parser.add_argument(
        "--set",
        dest="set_cfgs",
        help="set config keys",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument(
        "--dataset",
        dest="dataset",
        help="dataset name [posetrack, 3dpw]",
    )

    args = parser.parse_args()
    return args

def create_cfg() -> CfgNode:
    args = parse_args()
    if args.dataset == "posetrack":
        from utility.config_PT import get_cfg_defaults
    elif args.dataset == "3dpw":
        from utility.config_3dpw import get_cfg_defaults
    else:
        print("Please insert the correct dataset name!")
        sys.exit()
    cfg = get_cfg_defaults()
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
        experiment_name = args.cfg_file.replace('\\', '/')
    if args.set_cfgs is not None:
        cfg.merge_from_list(args.set_cfgs)
        experiment_name = args.set_cfgs.replace('\\', '/')
    cfg.freeze()
    return cfg, experiment_name.split('/')[-1].split('.yaml')[0]

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cfg, experiment_name = create_cfg()

    if (cfg.model.Msg_pass and not cfg.trainmode.human2obj) or (cfg.model.Msg_pass and not cfg.trainmode.use_social):
        print("!!!!!Message pathing mode can be applied only with both H2O and H2h being True!!!!!")
        exit()


    # Specify the directoy to save summaries (tensorboard events)
    summaries_dir = "./experiments/" +cfg.dataset.dataset_name+'/'+ experiment_name + '/summary'
    if not os.path.exists(summaries_dir):
        summaries_dir = summaries_dir + '/1'
        os.makedirs(summaries_dir)
    else:
        last_folder = np.max(list(map(int, os.listdir(summaries_dir))))
        if cfg.experiment.load == 0:
            if os.listdir(summaries_dir + '/' + str(last_folder)) == []:
                summaries_dir = summaries_dir + '/' + str(last_folder)
            else:
                summaries_dir = summaries_dir + '/' + str(last_folder + 1)
                os.makedirs(summaries_dir)
        else:
            summaries_dir = summaries_dir + '/' + str(last_folder)

    # Save the final configs
    if not os.path.exists(summaries_dir.split('summary')[0] + 'config/'):
        os.makedirs(summaries_dir.split('summary')[0] + 'config/')
    file1 = open(summaries_dir.split('summary')[0] + 'config/final_config_'+str(last_folder+1)+'.yaml', "w")  # append mode
    file1.write(cfg.dump())
    file1.close()

    #create the log folder
    if not os.path.exists(summaries_dir.split('summary')[0]+'logs/'):
        os.makedirs(summaries_dir.split('summary')[0]+'logs/')

    if cfg.experiment.load == 1:
        cfg.defrost()
        ckpt_path = summaries_dir.split('summary')[0] + '/checkpoints/' + summaries_dir.split('summary')[1] + '/best/'
        cfg.model.ckpt_path = ckpt_path + os.listdir(ckpt_path)[-1]
        cfg.freeze()

    main(cfg, summaries_dir)