import numpy as np
import pickle
import torch
import torch.nn as nn
import re
from typing import Tuple, Union, Set, List
from torch import Tensor

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def boolean(s):
    if s == '1':
        return True
    elif s == '0':
        return False
    msg = 'Invalid value "%s" for bool flag (should be 0 or 1)'
    raise ValueError(msg % s)

def get_dtypes(cfg):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if cfg.experiment.use_cpu == 0:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)

def Normalize_data(cfg, offsetData, poseData, complete_train_offset, complete_test_offset, complete_validation_offset, complete_train_pose, complete_test_pose, complete_validation_pose):
    if cfg.experiment.normalization == 'zscore':
        scaler = StandardScaler()
        scaler2 = StandardScaler()
    else:
        scaler = MinMaxScaler()
        scaler2 = MinMaxScaler()
    scaler = scaler.fit(complete_train_offset)
    scaled_data = scaler.transform(complete_train_offset)
    # scaled_data = scaler.fit_transform(complete_train_offset)
    # print(scaled_data.mean(axis=0))
    # print(scaled_data.std(axis=0)

    #Normalize data and reformat
    cc = 0
    train_set2 = []
    for i in range(len(offsetData['train'])):
        tmparr = np.zeros(offsetData['train'][i].shape)
        for p in range(offsetData['train'][i].shape[0]):
            for f in range(offsetData['train'][i].shape[1]):
                tmparr[p,f,:] = scaled_data[cc,:]
                cc += 1
        train_set2.append(tmparr)
    offsetData['train'] = train_set2

    scaled_data = scaler.transform(complete_test_offset)
    cc = 0
    test_set2 = []
    for i in range(len(offsetData['test'])):
        tmparr = np.zeros(offsetData['test'][i].shape)
        for p in range(offsetData['test'][i].shape[0]):
            for f in range(offsetData['test'][i].shape[1]):
                tmparr[p, f, :] = scaled_data[cc, :]
                cc += 1
        test_set2.append(tmparr)
    offsetData['test'] = test_set2

    scaled_data = scaler.transform(complete_validation_offset)
    cc = 0
    validation_set2 = []
    for i in range(len(offsetData['validation'])):
        tmparr = np.zeros(offsetData['validation'][i].shape)
        for p in range(offsetData['validation'][i].shape[0]):
            for f in range(offsetData['validation'][i].shape[1]):
                tmparr[p, f, :] = scaled_data[cc, :]
                cc += 1
        validation_set2.append(tmparr)
    offsetData['validation'] = validation_set2

    #create scaler for pose information
    scaler2 = scaler2.fit(complete_train_pose)
    scaled_data = scaler2.transform(complete_train_pose)
    cc = 0
    cp , cf, ci = 0,0,0
    PoseDatatrain2 = []
    for i in range(len(poseData['train'])):
        ci += 1
        tmparr = np.zeros(poseData['train'][i].shape)
        for p in range(poseData['train'][i].shape[0]):
            cp+=1
            for f in range(poseData['train'][i].shape[1]):
                cf+=1
                tmparr[p, f, :] = scaled_data[cc, :]
                cc += 1
        PoseDatatrain2.append(tmparr)
    poseData['train'] = PoseDatatrain2

    scaled_data = scaler2.transform(complete_test_pose)
    cc = 0
    PoseDatatest2 = []
    for i in range(len(poseData['test'])):
        tmparr = np.zeros(poseData['test'][i].shape)
        for p in range(poseData['test'][i].shape[0]):
            for f in range(poseData['test'][i].shape[1]):
                tmparr[p, f, :] = scaled_data[cc, :]
                cc += 1
        PoseDatatest2.append(tmparr)
    poseData['test'] = PoseDatatest2

    scaled_data = scaler2.transform(complete_validation_pose)
    cc = 0
    PoseDatavalidation2 = []
    for i in range(len(poseData['validation'])):
        tmparr = np.zeros(poseData['validation'][i].shape)
        for p in range(poseData['validation'][i].shape[0]):
            for f in range(poseData['validation'][i].shape[1]):
                tmparr[p, f, :] = scaled_data[cc, :]
                cc += 1
        PoseDatavalidation2.append(tmparr)
    poseData['validation'] =  PoseDatavalidation2

    return offsetData, poseData, scaler, scaler2

def Normalize_pose(cfg, poseData, complete_train_pose, complete_test_pose, complete_validation_pose):
    if cfg.experiment.normalization == 'zscore':
        scaler2 = StandardScaler()
    else:
        scaler2 = MinMaxScaler()

    #create scaler for pose information
    scaler2 = scaler2.fit(complete_train_pose)
    scaled_data = scaler2.transform(complete_train_pose)
    cc = 0
    cp , cf, ci = 0,0,0
    PoseDatatrain2 = []
    for i in range(len(poseData['train'])):
        ci += 1
        tmparr = np.zeros(poseData['train'][i].shape)
        for p in range(poseData['train'][i].shape[0]):
            cp+=1
            for f in range(poseData['train'][i].shape[1]):
                cf+=1
                tmparr[p, f, :] = scaled_data[cc, :]
                cc += 1
        PoseDatatrain2.append(tmparr)
    poseData['train'] = PoseDatatrain2

    scaled_data = scaler2.transform(complete_test_pose)
    cc = 0
    PoseDatatest2 = []
    for i in range(len(poseData['test'])):
        tmparr = np.zeros(poseData['test'][i].shape)
        for p in range(poseData['test'][i].shape[0]):
            for f in range(poseData['test'][i].shape[1]):
                tmparr[p, f, :] = scaled_data[cc, :]
                cc += 1
        PoseDatatest2.append(tmparr)
    poseData['test'] = PoseDatatest2

    scaled_data = scaler2.transform(complete_validation_pose)
    cc = 0
    PoseDatavalidation2 = []
    for i in range(len(poseData['validation'])):
        tmparr = np.zeros(poseData['validation'][i].shape)
        for p in range(poseData['validation'][i].shape[0]):
            for f in range(poseData['validation'][i].shape[1]):
                tmparr[p, f, :] = scaled_data[cc, :]
                cc += 1
        PoseDatavalidation2.append(tmparr)
    poseData['validation'] =  PoseDatavalidation2

    return poseData, scaler2

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, protocol=4)

def load_object(filename):
    with open(filename, 'rb') as input:  # Overwrites any existing file.
        return pickle.load(input)

def num2tensor(item):
    return torch.from_numpy(item).float()

def send_to(item, device):
    return item.to(device)

def repeat_per_row(item, rep_list):
    item1 = item.split(1)

    out = torch.FloatTensor([]).cuda()
    for x_sub, num_repeat in zip(item1, rep_list):
        out = torch.cat([out, x_sub.expand(num_repeat, -1)])
    return out

def get_object_features(objects_pickle, cfg, seq_name):
    dims = load_object('./data/' + cfg.dataset.dataset_name + '/seq_dimensions.pkl')
    image_dim = [x for x in dims if x[0] == seq_name.split('.json')[0]][0][1:]

    object = objects_pickle[list(objects_pickle.keys())[0]]  # frame15, 16th frame
    # objfeatures = np.zeros((np.count_nonzero(object['pred_classes']), len(object['rois'][0].flatten())), dtype=float)
    objfeatures = np.zeros((np.count_nonzero(object['pred_classes']), cfg.dataset.obj_dim),
                           dtype=float)
    if np.count_nonzero(object['pred_classes']) > 0:
        cnt = 0
        for i in range(len(object['pred_classes'])):   #for each object
            if object['pred_classes'][i] != 0:     #not a human
                x_lt = object['pred_boxes'][i][0]
                y_lt = object['pred_boxes'][i][1]
                x_rb = object['pred_boxes'][i][2]
                y_rb = object['pred_boxes'][i][3]
                centerX = x_lt + (x_rb - x_lt) / 2
                centerY = y_lt + (y_rb - y_lt) / 2
                h = (x_rb - x_lt)
                w = (y_rb - y_lt)
                #### objfeatures[cnt] = object['rois'][i].flatten()
                tmp = np.append(object['rois'][i].flatten(), [centerX/image_dim[1], centerY/image_dim[0], h/image_dim[1], w/image_dim[0]])
                objfeatures[cnt] = np.append(tmp, bitstring_to_array('{0:07b}'.format(object['pred_classes'][i]), cfg))
                # objfeatures[cnt] = np.append(object['rois'][0].flatten(), [centerX, centerY, h, w, object['pred_classes'][0]])
                cnt += 1

    return objfeatures

def bitstring_to_array(s, cfg):
    a = np.zeros(cfg.dataset.obj_lbl_size)
    matches = re.finditer('1', s)
    matches_positions = [match.start() for match in matches]
    a[matches_positions] = 1

    return a

def GetVelocity(poseData, offsetData):
    for key in ['train', 'test', 'validation']:
        cur_split = poseData[key]
        velocity3 = []
        for seq in range(len(cur_split)):
            returnArray = cur_split[seq]
            velocity2 = []
            for b in range(returnArray.shape[0]):
                velocity = []
                for i in range(returnArray.shape[1] - 1):
                    Cur = returnArray[b, i + 1]
                    Prev = returnArray[b, i]
                    veloc = Cur - Prev
                    velocity.append(veloc)

                velocity2.append(velocity)
            velocity2 = np.array(velocity2)
            velocity3.append(velocity2)
        offsetData[key] = velocity3

    return offsetData

    # def send_to(
#     items: Union[Tensor, Tuple[Tensor, ...], List[Tensor]], device
# ) -> Union[Tensor, Tuple[Tensor, ...], List[Tensor]]:
#     if type(items) is tuple:
#         return tuple(map(lambda x: x.to(device), items))
#     elif type(items) is list:
#         return list(map(lambda x: x.to(device), items))
#     else:
#         return items.to(device)