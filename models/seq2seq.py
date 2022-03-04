import torch
import torch.nn as nn
import numpy as np
from six.moves import xrange
from models.losses import l2_loss
from models.gat import GAT
from models.input_gat import inp_GAT
from models.human2obj import h2o_GAT
from utility.classes import LoggingPrinter
import utility.settings as settings
import utility.utils as utils

def init_step_curric(
        obs_pose, pred_pose_gt,  cfg, model, optimizer_g, forward_only, Mask_offset, Mask_pose, decoder_len, I3d_feature, obj_features
):
    """
    :param obs_pose:
    :param pred_pose_gt:
    :param cfg:
    :param model:
    :param optimizer_g:
    :param forward_only:
    :param Mask_offset:
    :param Mask_pose:
    :param decoder_len:
    :param I3d_feature:
    :param obj_features:
    :return:
    """

    losses = {}
    loss = torch.zeros(1).to(pred_pose_gt)
    l2_loss_val = []

    pred = model(obs_pose.to(torch.float32), len(pred_pose_gt), model.seq_start_end, I3d_feature, obj_features, model.seq_start_end_objects)

    if cfg.dataset.dataset_name == "posetrack":
        Mask_S = np.zeros((Mask_pose.shape[0], Mask_pose.shape[1], int(Mask_pose.shape[2] / 2))) + 1
        Mask_S = torch.from_numpy(Mask_S).cuda().to(torch.float32)
        l2_loss_val.append(l2_loss(pred, pred_pose_gt[0:decoder_len,:,:], Mask_offset[:,0:decoder_len,:], Mask_pose[:,0:decoder_len,:], Mask_S[:,0:decoder_len,:], cfg, mode='average'))
    else:
        l2_loss_val.append(l2_loss(pred, pred_pose_gt[0:decoder_len, :, :], Mask_offset,Mask_pose, [], cfg, mode='average'))

    losses['l2_loss'] = l2_loss_val[0].item()
    loss += l2_loss_val[0]

    Obs_Pred = torch.cat([obs_pose.to(torch.float32), pred], dim=0)

    if not forward_only:
        optimizer_g.zero_grad()
        loss.backward()
        if cfg.learning.clipping_threshold > 0:
            nn.utils.clip_grad_norm_(
                model.parameters(), cfg.learning.clipping_threshold
            )
        optimizer_g.step()

    return losses, Obs_Pred, pred

def make_mlp(dim_list, activation='leakyrelu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)

class Encoder(nn.Module):
    def __init__(
            self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1,
            dropout=0.0, Inp_size=0
    ):
        super(Encoder, self).__init__()

        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )

        self.spatial_embedding = nn.Linear(Inp_size, embedding_dim)

    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
            torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        )

    def forward(self, obs_data):
        """
        :param obs_data: Tensor of shape (obs_len, batch, joints)
        :return: final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        # Encode observation
        batch = obs_data.size(1)
        # obs_embedding = self.spatial_embedding(obs_data.view(-1, obs_data.shape[2]))
        obs_embedding = obs_data.contiguous().view(-1, obs_data.shape[2])
        obs_embedding = obs_embedding.view(-1, batch, self.embedding_dim)

        state_tuple = self.init_hidden(batch)
        output, state = self.encoder(obs_embedding, state_tuple)
        final_h = state[0]
        return final_h

class Decoder_Curriculum(nn.Module):
    def __init__(
            self, cfg, seq_len, embedding_dim=64, h_dim=128, mlp_dim=1024, num_layers=1, dropout=0.0, Inp_size=0, pool_every_timestep=True, pooling_type='gat', h2hmodel = []
    ):
        super(Decoder_Curriculum, self).__init__()

        self.seq_len = seq_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.pool_every_timestep = pool_every_timestep

        if pool_every_timestep:
            if pooling_type == 'gat':
                self.pooling = h2hmodel#GAT(cfg, self.h_dim)
            elif pooling_type == 'max':
                self.pooling = MaxPooling(
                    h_dim=self.h_dim
                )
            self.batchN = nn.BatchNorm1d(self.pooling.h_dim)
            mlp_dims = [h_dim, mlp_dim]
            self.mlp_beforeh2h = make_mlp(
                mlp_dims,
                activation='leakyrelu',
                batch_norm=False,
                dropout=dropout
            )

            mlp_dims = [h_dim + self.pooling.h_dim, mlp_dim, h_dim]
            self.mlp = make_mlp(
                mlp_dims,
                activation='leakyrelu',
                batch_norm=False,
                dropout=dropout
            )


        self.decoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )

        self.spatial_embedding = nn.Linear(Inp_size, embedding_dim)
        self.hidden2pose = nn.Linear(h_dim, Inp_size)

    def forward(self, last_pos,  state_tuple, decoder_len, residual, seq_start_end, context_features):
        """
        :param last_pos: Tensor of shape (batch, joints)
               state_tuple: (hh, ch) each tensor of shape (num_layers, batch, h_dim)
               decoder_len: current lenght of prediction frames in current step of curriculum learning
               residual: if true use skip connection between input and output of decoder
        :return: pred_pose: tensor of shape (self.seq_len, batch, joints)
        """
        batch = last_pos.size(0)
        pred_pose = []
        decoder_input = last_pos
        # decoder_input = self.spatial_embedding(last_pos)  # (last_pos_rel)
        decoder_input = decoder_input.view(1, batch, self.embedding_dim)

        for _ in range(decoder_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            real_pose = self.hidden2pose(output.view(-1, self.h_dim))
            if residual == True:
                real_pose = torch.add(real_pose.view(1, batch, last_pos.size(1)),decoder_input)
                real_pose = real_pose.view(batch, last_pos.size(1))
            # curr_pos = rel_pos + last_pos

            if self.pool_every_timestep:
                decoder_h = self.mlp_beforeh2h(state_tuple[0])
                pool_h = self.pooling(decoder_h, seq_start_end)
                pool_h = self.batchN(pool_h)
                decoder_h = torch.cat([state_tuple[0].view(-1, self.h_dim), pool_h], dim=1)
                decoder_h = self.mlp(decoder_h)
                decoder_h = torch.unsqueeze(decoder_h, 0)
                state_tuple = (decoder_h, state_tuple[1])

            embedding_input = real_pose
            # decoder_input = self.spatial_embedding(embedding_input)
            decoder_input = embedding_input
            decoder_input = decoder_input.view(1, batch, self.embedding_dim)
            pred_pose.append(real_pose.view(batch, -1))
            # last_pos = curr_pos

        pred_pose = torch.stack(pred_pose, dim=0)
        return pred_pose, state_tuple[0]


class Seq2SeqModel_Curriculum(nn.Module):
    """Sequence-to-sequence model for human motion and pose prediction.
    params:
        shapedata: shape of input features
        cfg: input parameters
    """

    def __init__(
            self,
            shapedata,
            cfg
    ):
        super(Seq2SeqModel_Curriculum, self).__init__()

        self.HUMAN_SIZE = shapedata
        self.Add_offset = cfg.trainmode.Add_offset
        self.Add_pose = cfg.trainmode.Add_pose
        self.use_social = cfg.trainmode.use_social
        self.cfg = cfg

        if cfg.trainmode.Add_visib == True and cfg.trainmode.Add_offset == True and cfg.trainmode.Add_pose == True:
            # two feature vector of size 28 for offset and pose and one with size 14 for visibilities
            self.input_size = self.HUMAN_SIZE * 2 + (int(self.HUMAN_SIZE / 2))
            self.jsize = int((self.input_size - int(self.input_size / 5)) / 2)
            self.node_hsize = (cfg.dataset.dim * 2) + 1
        elif cfg.trainmode.Add_visib == False and cfg.trainmode.Add_offset == True and cfg.trainmode.Add_pose == True:
            self.input_size = self.HUMAN_SIZE * 2
            self.jsize = int(self.input_size / 2)
            self.node_hsize = (cfg.dataset.dim * 2)
        elif cfg.trainmode.Add_visib == False and cfg.trainmode.Add_offset == False and cfg.trainmode.Add_pose == True:
            self.input_size = self.HUMAN_SIZE
            self.jsize = int(self.input_size)
            self.node_hsize = cfg.dataset.dim
        elif cfg.trainmode.Add_visib == False and cfg.trainmode.Add_offset == True and cfg.trainmode.Add_pose == False:
            self.input_size = self.HUMAN_SIZE
            self.jsize = int(self.input_size)
            self.node_hsize = cfg.dataset.dim


        if cfg.experiment.load == 0:
            with LoggingPrinter(settings.log_file): print("Input size is %d" % self.input_size)

        self.source_seq_len = cfg.model.seq_length_in
        self.target_seq_len = cfg.model.seq_length_out
        self.batch_size = cfg.learning.batch_size
        # self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=dtype)
        # self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        # self.global_step = tf.Variable(0, name='global_step', trainable=False)
        # self.use_context = use_context
        self.encoder_h_dim = cfg.model.encoder_h_dim
        self.num_layers = cfg.model.num_layers
        self.injectLayer_num = cfg.model.injectLayer_num
        self.residual_velocities = cfg.model.residual_velocities
        self.decoder_h_dim = cfg.model.decoder_h_dim
        self.pool_every_timestep = cfg.model.pool_every_timestep
        self.input_representation = cfg.model.input_representation
        self.GAT_residual = cfg.model.GAT_res

        self.input_size_g = self.input_size
        if cfg.model.input_representation == 'graph':
            self.inputGat = inp_GAT(self.cfg, cfg.model.inp_graph_outdim, self.node_hsize)
            if self.GAT_residual:
                self.input_size_g = (int(self.input_size / self.node_hsize) * cfg.model.inp_graph_outdim) + self.input_size
            else:
                self.input_size_g = int(self.input_size/self.node_hsize) * cfg.model.inp_graph_outdim

        self.encoder = Encoder(
            embedding_dim=self.input_size_g,
            h_dim=self.encoder_h_dim,
            mlp_dim=cfg.model.mlp_dim,
            num_layers=self.num_layers,
            dropout=cfg.model.dropout,
            Inp_size=self.input_size_g
        )

        if cfg.trainmode.human2obj == True:
            mlp_dims = [cfg.dataset.obj_dim, 5000, 1024, self.encoder_h_dim]
            self.obj_mlp = make_mlp(
                mlp_dims,
                activation='leakyrelu',
                batch_norm=False,
                dropout=cfg.model.object_dropout
            )
            self.obj_batchN = nn.BatchNorm1d(self.encoder_h_dim)
            self.h2o_graph = h2o_GAT(self.cfg, self.encoder_h_dim)
            self.h2o_batchN = nn.BatchNorm1d(self.encoder_h_dim)

        if self.use_social == True:
            if cfg.model.pool_type == 'gat':
                self.pooling = GAT(self.cfg, self.encoder_h_dim)
                self.decoder_h_dim = self.encoder_h_dim + self.encoder_h_dim
            self.h2h_batchN = nn.BatchNorm1d(self.encoder_h_dim)

        if cfg.trainmode.use_context == True:
            mlp_dims = [cfg.dataset.context_dim, 512, cfg.model.context_fc2_size]
            self.context_mlp = make_mlp(
                mlp_dims,
                activation='leakyrelu',
                batch_norm=False,
                dropout=cfg.model.context_dropout
            )
            self.decoder_h_dim = 2*self.encoder_h_dim + cfg.model.context_fc2_size#+ self.encoder_h_dim + cfg.model.context_fc2_size
            self.context_batchN = nn.BatchNorm1d(cfg.model.context_fc2_size)

        if self.pool_every_timestep == False:
            self.decoder = Decoder_Curriculum(
                self.cfg,
                self.target_seq_len,
                embedding_dim=self.input_size,
                h_dim=self.decoder_h_dim,
                mlp_dim=cfg.model.mlp_dim,
                num_layers=self.num_layers,
                dropout=cfg.model.dropout,
                Inp_size=self.input_size,
                pool_every_timestep=self.pool_every_timestep
            )
        else:
            self.decoder = Decoder_Curriculum(
                self.cfg,
                self.target_seq_len,
                embedding_dim=self.input_size,
                h_dim=self.decoder_h_dim,
                mlp_dim=cfg.model.mlp_dim,
                num_layers=self.num_layers,
                dropout=cfg.model.dropout,
                Inp_size=self.input_size,
                pool_every_timestep=self.pool_every_timestep,
                h2hmodel=self.pooling
            )

    def forward(self, obs_pose, decoder_len, seq_start_end, I3d_feature, obj_features, seq_start_end_objects):
        """
        :param obs_pose: Tensor of shape (obs_len, batch, joints)
               decoder_len: current lenght of decoder in curriculum learning
               seq_start_end: A list of tuples which delimit sequences within batch.
        :return: pred_pose: Tensor of shape (pred_len, batch, joints)
        """
        batch = obs_pose.size(1)
        #Input space
        graph_input = self.inputGat(obs_pose, seq_start_end)
        if self.GAT_residual:
            graph_input = torch.cat([obs_pose.contiguous().view(-1, obs_pose.shape[2]), graph_input.view(-1, graph_input.shape[2])], dim=1)
            graph_input = graph_input.view(obs_pose.shape[0], obs_pose.shape[1], -1)
        # Encode seq
        final_encoder_h = self.encoder(graph_input)


        obj_features = self.obj_mlp(obj_features)
        obj_features = self.obj_batchN(obj_features)
        h2o_inp = final_encoder_h
        for iter in range(self.cfg.model.Num_msg_pass):
                pool_h2o = self.h2o_graph(h2o_inp, obj_features, seq_start_end, seq_start_end_objects)
                pool_h2o = self.h2o_batchN(pool_h2o)
                pool_h2o = torch.add(pool_h2o, h2o_inp.view(-1, self.encoder_h_dim))

                pool_h2h = self.pooling(pool_h2o, seq_start_end)
                pool_h2h = self.h2h_batchN(pool_h2h)
                mlp_decoder_context_input = torch.add(pool_h2h, pool_h2o)

                h2o_inp = mlp_decoder_context_input

                if iter == self.cfg.model.Num_msg_pass-1:
                    mlp_decoder_context_input = torch.cat([mlp_decoder_context_input.view(-1, self.encoder_h_dim), final_encoder_h.view(-1, self.encoder_h_dim)],dim=1)



        decoder_h = torch.unsqueeze(mlp_decoder_context_input, 0)
        decoder_c = torch.zeros(self.num_layers, batch, self.decoder_h_dim).cuda()


        if self.cfg.trainmode.use_context == True:
            I3d_feature = self.context_mlp(I3d_feature)
            I3d_feature = self.context_batchN(I3d_feature)  #**batch****
            I3d_feature = utils.repeat_per_row(I3d_feature, list(np.array(seq_start_end)[:, 1] - np.array(seq_start_end)[:, 0]))
            decoder_h = torch.unsqueeze(torch.cat([decoder_h.view(-1, self.decoder_h_dim - self.cfg.model.context_fc2_size), I3d_feature], dim=1), 0)


        state_tuple = (decoder_h, decoder_c)
        last_pos = obs_pose[-1]

        decoder_out = self.decoder(
            last_pos,
            state_tuple,
            decoder_len,
            self.residual_velocities,
            seq_start_end,
            I3d_feature
        )
        pred_pose, final_decoder_h = decoder_out

        return pred_pose

    def get_batch(self, offset_data, current_step, PoseDatatrain, cfg, S_Datatrain, fileNames, MissMaskoffset, MissMaskpose, injection_step):

        # Select entries
        # chosen_keys = np.random.choice(len(offset_data), self.batch_size, replace=False)
        chosen_keys = list(range((current_step)*self.batch_size,(current_step+1)*self.batch_size))

        #if current batch does not exceed data size
        if (current_step+1)*self.batch_size > len(offset_data):
            bsize = len(offset_data) - chosen_keys[0]
        else:
            bsize = self.batch_size

        num_persons = 0
        num_peds_in_seq = []
        for i in xrange(bsize):
            num_persons = num_persons + PoseDatatrain[chosen_keys[i]].shape[0]
            num_peds_in_seq.append(PoseDatatrain[chosen_keys[i]].shape[0])
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

        encoder_inputs = np.zeros((num_persons, self.source_seq_len, self.input_size), dtype=float)
        decoder_outputs = np.zeros((num_persons, injection_step, self.input_size), dtype=float)
        i3d_features = np.zeros((bsize, cfg.dataset.context_dim), dtype=float)
        obj_features = [] #np.zeros((bsize, cfg.dataset.obj_dim), dtype=float)
        self.seq_start_end_objects = []

        gt_pose_positions = np.zeros((num_persons, injection_step, self.jsize), dtype=float)
        last_seen_pose = np.zeros((num_persons, 1, self.jsize), dtype=float)
        Mask_offset = np.zeros((num_persons, injection_step, self.jsize), dtype=float)
        Mask_pose = np.zeros((num_persons, injection_step, self.jsize), dtype=float)

        idx = 0
        obj_idx = 0
        for i in xrange(bsize):
            if cfg.trainmode.use_context == True and cfg.dataset.dataset_name == 'posetrack':
                i3d_features[i] = np.load(cfg.dataset.I3d_features_path + '/' + fileNames[chosen_keys[i]].split('.json')[0] + '/joint/i3dfeature.npy',
                    allow_pickle=True)

            if cfg.trainmode.use_context == True and cfg.dataset.dataset_name == '3dpw':
                split_idx = np.load(cfg.dataset.split_data_path + '/split_idx_train.pkl', allow_pickle=True)
                i3d_features[i] = np.load(cfg.dataset.I3d_features_path + '/' + fileNames[chosen_keys[i]] + '_' + str(split_idx[chosen_keys[i]]) + '/joint/i3dfeature.npy',
                    allow_pickle=True)

            if cfg.trainmode.human2obj == True:
                if cfg.dataset.dataset_name == '3dpw':
                    split_idx = np.load(cfg.dataset.split_data_path + '/split_idx_train.pkl', allow_pickle=True)
                    obj_features, obj_idx = self.get_obj_features(cfg, fileNames[chosen_keys[i]] + '_' + str(split_idx[chosen_keys[i]]), obj_features, obj_idx, fileNames[chosen_keys[i]])
                else:
                    obj_features, obj_idx = self.get_obj_features(cfg, fileNames[chosen_keys[i]].split('.json')[0], obj_features, obj_idx, fileNames[chosen_keys[i]])



            for p in range(offset_data[chosen_keys[i]].shape[0]):
                st = 0
                # Select the offset_data around the sampled points
                if self.Add_offset == True:
                    data_sel = offset_data[chosen_keys[i]]
                    # Add the offset_data
                    encoder_inputs[idx, :, st:self.jsize] = data_sel[p, 0:self.source_seq_len, :]
                    decoder_outputs[idx, 0:injection_step, st:self.jsize] = data_sel[p, self.source_seq_len:self.source_seq_len + injection_step,0:self.jsize]
                    st = st+self.jsize

                if cfg.dataset.dataset_name == 'posetrack':
                    data_sel = MissMaskoffset[chosen_keys[i]]
                    Mask_offset[idx, 0:injection_step, 0:self.jsize] = data_sel[p, self.source_seq_len:self.source_seq_len + injection_step,0:self.jsize]
                    data_sel = MissMaskpose[chosen_keys[i]]
                    Mask_pose[idx, 0:injection_step, 0:self.jsize] = data_sel[p, self.source_seq_len + 1:self.source_seq_len + injection_step + 1,0:self.jsize]

                # Select the pose_data around the sampled points
                if self.Add_pose == True:
                    pose_sel = PoseDatatrain[chosen_keys[i]]
                    encoder_inputs[idx, :, st:st+self.jsize] = pose_sel[p, 1:self.source_seq_len + 1, :]
                    decoder_outputs[idx, 0:injection_step, st:st + self.jsize] = pose_sel[p, self.source_seq_len + 1:self.source_seq_len + injection_step + 1,0:self.jsize]
                    st = st+self.jsize

                gt_pose_positions[idx, 0:injection_step, :] = pose_sel[p, self.source_seq_len + 1:self.source_seq_len + injection_step + 1,:]
                last_seen_pose[idx, :, :] = pose_sel[p, self.source_seq_len, :]

                if cfg.trainmode.Add_visib == True and cfg.dataset.dataset_name == 'posetrack':
                    s_sel = S_Datatrain[chosen_keys[i]]
                    encoder_inputs[idx, :, st:] = s_sel[p, 1:self.source_seq_len + 1, :]
                    decoder_outputs[idx, :, st:] = s_sel[p, self.source_seq_len + 1:self.source_seq_len + injection_step + 1, :]

                idx += 1

        if obj_features != []:
            obj_features = np.array(np.concatenate(obj_features, axis=0))
        else:
            obj_features = np.array(obj_features)
        return encoder_inputs, decoder_outputs, gt_pose_positions, last_seen_pose, Mask_offset, Mask_pose, i3d_features, obj_features

    def get_batch_test(self, offsetData, seq, PoseData, cfg, visibData, MissMaskoffset, MissMaskpose, injection_step):
        """ Get a batch of data for test, prepare for step. here batchsize=#personInSeq, processing each seuence one by one
                :param offsetData: list of offset information, shape: ({#seq}(PERSONxF-1xself.jsize))
                       seq: current step of testing/validating
                       PoseData: list of poselocation information, shape: ({#seq}(PERSONxFxself.jsize))
                       cfg: input parameters
                       visibData: list of joint visibility information. shape: ({#seq}(PERSONxFxself.jsize/2))
                       MissMaskoffset: list of visiblity masks for offset. shape: ({#seq}(PERSONxF-1xself.jsize))
                       MissMaskpose: list of visiblity masks for poselocation. shape: ({#seq}(PERSONxFxself.jsize))
                       injection_step: current step in curriculum learning
                :return: encoder_inputs: array of shape(#personInSeq,obs_len,self.input_size)
                    decoder_outputs: array of shape(#personInSeq,pred_len,self.input_size)
                    gt_pose_positions: ground thruth of poses, array of shape(#personInSeq,pred_len,self.jsize)
                    last_seen_pose: last observed pose, array of shape(#personInSeq,1,self.jsize)
                    Mask_offset: visibility mask of offset information for prediction, array of shape(#personInSeq,pred_len,self.jsize)
                    Mask_pose: visibility mask of pose information for prediction, array of shape(#personInSeq,pred_len,self.jsize)
                    Mask_pose_obs: visibility mask of pose information for observation, array of shape(#personInSeq,obs_len,self.jsize)
                """
        #test one by one
        batchsize = 1  # self.batch_size

        chosen_keys = seq

        # Get the number of person in current sequence
        PeopleNum, _, _ = offsetData[chosen_keys].shape

        num_peds_in_seq = []
        num_peds_in_seq.append(PeopleNum)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

        encoder_inputs = np.zeros((PeopleNum, self.source_seq_len, self.input_size), dtype=float)
        decoder_outputs = np.zeros((PeopleNum, injection_step, self.input_size), dtype=float)

        gt_pose_positions = np.zeros((PeopleNum, injection_step, self.jsize), dtype=float)
        last_seen_pose = np.zeros((PeopleNum, 1, self.jsize), dtype=float)
        Mask_offset = np.zeros((PeopleNum, injection_step, self.jsize), dtype=float)
        Mask_pose = np.zeros((PeopleNum, injection_step, self.jsize), dtype=float)
        Mask_pose_obs = np.zeros((PeopleNum, self.source_seq_len, self.jsize), dtype=float)

        for p in range(PeopleNum):
            st = 0
            if self.Add_offset == True:
                # Select the data around the sampled points
                data_sel = offsetData[chosen_keys][p]
                # Add the offsetData
                encoder_inputs[p, :, st:self.jsize] = data_sel[0:self.source_seq_len, :]
                decoder_outputs[p, :, st:self.jsize] = data_sel[self.source_seq_len:self.source_seq_len +injection_step,0:self.jsize]
                st = st + self.jsize

            if cfg.dataset.dataset_name == 'posetrack':
                data_sel = MissMaskoffset[chosen_keys][p]
                Mask_offset[p, :, 0:self.jsize] = data_sel[self.source_seq_len:self.source_seq_len + injection_step,0:self.jsize]
                data_sel = MissMaskpose[chosen_keys][p]
                Mask_pose[p, :, 0:self.jsize] = data_sel[self.source_seq_len + 1:self.source_seq_len + injection_step + 1,0:self.jsize]
                Mask_pose_obs[p, :, 0:self.jsize] = data_sel[1: self.source_seq_len + 1]

            if self.Add_pose == True:
                pose_sel = PoseData[chosen_keys][p]
                encoder_inputs[p, :, st:st+self.jsize] = pose_sel[1:self.source_seq_len + 1, :]
                decoder_outputs[p, :, st:st+self.jsize] = pose_sel[self.source_seq_len + 1:self.source_seq_len + injection_step + 1, 0:self.jsize]
                st = st + self.jsize

            gt_pose_positions[p, :, :] = pose_sel[self.source_seq_len + 1:self.source_seq_len + injection_step + 1,:]
            last_seen_pose[p, :, :] = pose_sel[self.source_seq_len, :]

            if cfg.trainmode.Add_visib == True and cfg.dataset.dataset_name == 'posetrack':
                s_sel = visibData[chosen_keys][p]
                encoder_inputs[p, :, st:] = s_sel[1:self.source_seq_len +1, :]
                decoder_outputs[p, :, st:] = s_sel[self.source_seq_len + 1:self.source_seq_len + injection_step + 1,:]

        return encoder_inputs, decoder_outputs, gt_pose_positions, last_seen_pose, Mask_offset, Mask_pose, Mask_pose_obs

    def get_obj_features(self, cfg, fileNames, obj_features, obj_idx, seq_name):
        object = utils.load_object('./' + cfg.dataset.obj_features_path + '/' + fileNames + '/detections.pkl')
        obj = utils.get_object_features(object, cfg, seq_name)

        if obj.shape[0] > 0:
            obj_features.append(obj)
        self.seq_start_end_objects.append((obj_idx, obj_idx + obj.shape[0]))
        obj_idx += obj.shape[0]

        # seq_dimensions = []
        # path = 'I:/Projects/HPF/Datasets/3DPW/imageFiles/'
        # import os
        # import cv2
        # sequences = os.listdir(path)
        # for seq in range(len(sequences)):
        #     img_path = path + sequences[seq] + '/' + os.listdir(path + sequences[seq])[0]
        #     img = cv2.imread(img_path)
        #     seq_dimensions.append([sequences[seq], img.shape[0], img.shape[1]])
        #     pp=1

        return obj_features, obj_idx


