import os
import cv2
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import copy

from utility.utils import save_object


selected_joints = [1,2,4,5,7,8,12,16,17,18,19,20,21]

def readSkeleton_positions(mode, cfg):
    pose_dim = cfg.dataset.dim #x, Y, Z
    datasetDir = cfg.dataset.skeleton_dir + mode
    fileNames = os.listdir(datasetDir)
    bodyinfo = []
    Final_filenames = []
    for i in range(len(fileNames)):  #for each file
        curfile = os.path.join(datasetDir, fileNames[i])
        with open(curfile, 'rb') as f:
            tmp = pkl.load(f, encoding='latin1')

        jointPositions = tmp['jointPositions']
        seq_bodies = np.zeros((len(jointPositions), jointPositions[0].shape[0], len(selected_joints)*pose_dim))
        if len(jointPositions) == 1:
            continue
        for p in range(len(jointPositions)):
            for f in range(jointPositions[p].shape[0]):
                curpose_p_f = jointPositions[p][f]
                curbody = []
                for j in range(int(len(curpose_p_f)/pose_dim)):
                    if j in selected_joints:
                        curbody.append(curpose_p_f[j * pose_dim])
                        curbody.append(curpose_p_f[j * pose_dim + 1])
                        curbody.append(curpose_p_f[j * pose_dim + 2])

                curbody = np.array(curbody)
                seq_bodies[p, f, :] = curbody

        bodyinfo.append(seq_bodies)
        Final_filenames.append(fileNames[i].split('.')[0])
    return bodyinfo, Final_filenames


def split_longSequence(action_sequence, fileNames, cfg, mode):
    skip = cfg.dataset.dataset_skip
    start_from = 20

    seq_len = cfg.model.seq_length_in + cfg.model.seq_length_out + 1

    Final_new_seq = []
    new_filename = []
    start_ends = []
    split_idx = []
    for seq in range(len(action_sequence)):
        seq_numbers = int((action_sequence[seq].shape[1]- start_from)/seq_len/skip)
        new_seq2 = []
        for i in range(seq_numbers):
            split_idx.append(i)
            new_seq = []
            new_filename.append(fileNames[seq])
            start = list(range((i * skip) * seq_len + start_from, (i + 1) * skip * seq_len + start_from, skip))[0]
            end = list(range((i * skip) * seq_len + start_from, (i + 1) * skip * seq_len + start_from, skip))[-1]
            start_ends.append((start, end))
            for b in range(action_sequence[seq].shape[0]):
                cur_seq_b = action_sequence[seq][b, :, :]

                new_seq.append(cur_seq_b[list(range((i*skip)*seq_len++ start_from,(i+1)*skip*seq_len++ start_from, skip)),:])
            new_seq2.append(np.array(new_seq))
        Final_new_seq = Final_new_seq + new_seq2

    # save_object(new_filename, './data/3dpw/splited_seq/seqNames_' + mode + '.pkl')
    # save_object(start_ends, './data/3dpw/splited_seq/start_ends_' + mode + '.pkl')
    # save_object(split_idx, './data/3dpw/splited_seq/split_idx_' + mode + '.pkl')

    return Final_new_seq, new_filename, start_ends, split_idx

def split_longSequence_withoverlap(action_sequence, fileNames, cfg, mode):
    skip = cfg.dataset.dataset_skip
    start_from = 20
    strike = cfg.dataset.strike

    seq_len = cfg.model.seq_length_in + cfg.model.seq_length_out + 1

    Final_new_seq = []
    new_filename = []
    start_ends = []
    split_idx = []
    if mode == 'train' or mode == 'validation':
        for seq in range(len(action_sequence)):

            seq_numbers = int(((action_sequence[seq].shape[1] - start_from)- (seq_len * skip))/strike) +1
            new_seq2 = []
            for i in range(seq_numbers):
                split_idx.append(i)
                new_seq = []
                new_filename.append(fileNames[seq])
                start = i * strike + start_from
                end = start + (skip * seq_len)
                idx = list(range(start, end, 2))

                start_ends.append((start, end))
                for b in range(action_sequence[seq].shape[0]):
                    cur_seq_b = action_sequence[seq][b, :, :]
                    new_seq.append(cur_seq_b[idx, :])
                new_seq2.append(np.array(new_seq))
            Final_new_seq = Final_new_seq + new_seq2
    else:
        for seq in range(len(action_sequence)):
            seq_numbers = int((action_sequence[seq].shape[1]- start_from)/seq_len/skip)
            new_seq2 = []
            for i in range(seq_numbers):
                split_idx.append(i)
                new_seq = []
                new_filename.append(fileNames[seq])
                start = list(range((i * skip) * seq_len + start_from, (i + 1) * skip * seq_len + start_from, skip))[0]
                end = list(range((i * skip) * seq_len + start_from, (i + 1) * skip * seq_len + start_from, skip))[-1]
                start_ends.append((start, end))
                for b in range(action_sequence[seq].shape[0]):
                    cur_seq_b = action_sequence[seq][b, :, :]
                    new_seq.append(cur_seq_b[list(range((i*skip)*seq_len++ start_from,(i+1)*skip*seq_len++ start_from, skip)),:])
                new_seq2.append(np.array(new_seq))
            Final_new_seq = Final_new_seq + new_seq2


    # save_object(new_filename, './data/3DPW/splited_seq/seqNames_' + mode + '_strike' +str(strike)+ '.pkl')
    # save_object(start_ends, './data/3DPW/splited_seq/start_ends_' + mode + '_strike' +str(strike)+ '.pkl')
    # save_object(split_idx, './data/3DPW/splited_seq/split_idx_' + mode + '_strike' +str(strike)+ '.pkl')

    return Final_new_seq, new_filename, start_ends, split_idx

def load_data(mode, cfg):

    bodyinfos, fileNames = readSkeleton_positions(mode, cfg)

    action_sequence = bodyinfos
    action_sequence, fileNames, start_ends, split_idx = split_longSequence(action_sequence, fileNames, cfg, mode)

    PoseData = copy.deepcopy(action_sequence)

    d = PoseData[0].shape[2]


    completeDataPose = []
    # [completeDataPose.extend(list(a[:,1:,:])) for a in PoseData]
    [completeDataPose.extend(list(a[:, :, :])) for a in PoseData]
    completeDataPose2 = []
    for b in range(len(completeDataPose)):
        for f in range(completeDataPose[b].shape[0]):
            completeDataPose2.append(completeDataPose[b][f, :])

    return PoseData, d, fileNames, np.array(completeDataPose2), split_idx

