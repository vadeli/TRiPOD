from yacs.config import CfgNode as CN
import os

_C = CN()

_C.system = CN()
_C.system.device = "cuda"
_C.system.num_workers = 0
_C.system.seed = 2

_C.learning = CN()
# Learning rate
_C.learning.learning_rate = .000005
# Learning rate is multiplied by this much. 1 means no decay.
_C.learning.learning_rate_decay_factor = 0.95
# Every this many steps, do decay.
_C.learning.learning_rate_step = 1000
# Clip gradients to this norm.
_C.learning.clipping_threshold = 5
# Batch size to use during training.
_C.learning.batch_size = 16
# Iterations to train for.
_C.learning.iterations = int(2e5)
# Epochs to train for.
_C.learning.epoch_num = int(600)
# Shuffle the training data
_C.learning.shuffle = 0

_C.model = CN()
# hidden size of encoder
_C.model.encoder_h_dim = 256
# hidden size of decoder
_C.model.decoder_h_dim = 256
# Number of layers in the model.
_C.model.num_layers = 1
# Number of frames to feed into the encoder. 25 fps
_C.model.seq_length_in = 16
# Number of frames that the decoder has to predict. 25fps
_C.model.seq_length_out = 14
# Add a residual connection that effectively models velocities
_C.model.residual_velocities = True
# Steps for injecting the new frame in curriculum (== #seq_length_out: simple training)
_C.model.injectLayer_num = 14
# dropout value for model
_C.model.dropout = 0
# dimension of mlp in model
_C.model.mlp_dim = 256
# Type of pooling used for social module.
_C.model.pool_type = "gat"
# Number of gat layers
_C.model.gat_layers = 1
# Number of gat heads
_C.model.gat_heads = 3
# The value for gat dropout
_C.model.gat_dropout = 0.5
# Perform social pooling every timestep in decoder
_C.model.pool_every_timestep = False
# The type of input representation: [tensor, graph]
_C.model.input_representation = "tensor"
#Graph connection type in input GAT: [FC, sparse]
_C.model.input_Graph_type = "FC"
# The output dim of the input graph
_C.model.inp_graph_outdim = 10
# checkpoint path to load
_C.model.ckpt_path = os.path.normpath("./")
# Output dimesion of context feature after feeding to MLP
_C.model.context_fc2_size = 128
# Dropout for context MLP
_C.model.context_dropout = 0.7
# Dropout for object MLP
_C.model.object_dropout = 0.5
# Loss for visibility features (MSE/BCE)
_C.model.visib_loss = 'BCE'
# alpha param of GAT model
_C.model.gat_alpha = 0.2
# Number of gat heads human to object
_C.model.h2o_gat_heads = 3
# Add a residual connection after GAT
_C.model.GAT_res = False
#Use message passing
_C.model.Msg_pass = True
#Iterations is message passing procedure
_C.model.Num_msg_pass = 4
#h2h/h2o as the first graph
_C.model.First_graph = 'h2o'

_C.trainmode = CN()
# Whether to use social pooling or not
_C.trainmode.use_social = True
# Whether to use scene context or not
_C.trainmode.use_context = True
# Whether to use pose offsets or not
_C.trainmode.Add_offset = True
# Whether to use pose locations or not
_C.trainmode.Add_pose = True
# Whether to use visibilities or not
_C.trainmode.Add_visib = False
# Whether to add trajectory at the end of input features
_C.trainmode.traj = 0
# Whether to center the pose or not
_C.trainmode.centered = 0
# human to object graph
_C.trainmode.human2obj = True


_C.experiment = CN()
# Whether to use the CPU
_C.experiment.use_cpu = False
# Try to load a previous checkpoint.
_C.experiment.load = 0
# How often to compute error on the test set.
_C.experiment.test_every = 20
# How often to compute error on the test set.
_C.experiment.save_every = 1000
# Print results evey this step
_C.experiment.print_every = 50
# Set to True for sampling.
_C.experiment.sample = False
#cutoff value for occlusion handling metric
_C.experiment.occ_cutoff = 200
#Data normalization_mode ('zscore', 'unit')
_C.experiment.normalization = 'zscore'


_C.dataset = CN()
_C.dataset.dataset_name = '3dpw'
# Training model directory
_C.dataset.train_dir = os.path.normpath("./experiments/")
# Skeleton annotation's directory.
_C.dataset.skeleton_dir = os.path.normpath("./data/3dpw/annotations/")
# Images of dataset directory
_C.dataset.Image_dir = os.path.normpath("./Datasets/3DPW/imageFiles/")
# I3D features path
# I3D features path
_C.dataset.I3d_features_path = os.path.normpath("./data/3dpw/kinetics-i3d/features/")
# object features path
_C.dataset.obj_features_path = os.path.normpath("./data/3dpw/objects/features/")
# object features path
_C.dataset.split_data_path = os.path.normpath("./data/3dpw/splited_seq/")
# cm to meter
_C.dataset.W_Scale = 100
# Dimension of context features extracted from I3D
_C.dataset.context_dim = 1024
# Dimension of object features extracted from Faster-RCNN
_C.dataset.obj_dim = 12544+5 #12544+4+7
#object class labelsize
_C.dataset.obj_lbl_size = 7
# Number of frames that we have to skip
_C.dataset.dataset_skip = 2
# Number of frames that we have to skip
_C.dataset.strike = 20
#dimension of pose (2d, 3d)
_C.dataset.dim = 3
#joints adjacency
_C.dataset.bones = list({(0,1), (0,2), (2,4), (1,3), (3,5), (6,0), (6,1), (6,7), (6,8), (7,9), (7,11), (8,10), (10,12)})


def get_cfg_defaults():
    return _C.clone()
