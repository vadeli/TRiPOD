import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models import seq2seq_posetrack
from models import Errors
from utility import Visualization, data_utils_3DPW
from utility.classes import LoggingPrinter
import utility.settings as settings
from utility.utils import num2tensor, send_to, load_object, get_object_features
from utility.occlusion_metric import *

def computeMetricForValidation_curric(cfg, model, offsetData_val, PoseDatavalidation, Val_metricVal, All_val, T_val_loss, validation_set_fileNames,current_step, writer, S_data, offset_miss, pose_miss, scaler_pose, optimizer_g, injection_step):
    """ forward validation data seq by seq and compute metric
    :param model:trained seq2seq model
         offsetData_val: list of array of offset information for validation sequences {#seq}(#person x F-1 x model.jsize)
         PoseDatavalidation: list of array of pose locations for validation sequences {#seq}(#person x F x model.jsize)
         Val_metricVal: list of metric value of previous last time step
         All_val: list of metric value of previous all time step
         T_val_loss: list of previous step validation losses
         validation_set_fileNames: file names of validation split
         current_step: total current step of training
         writer: the writer used for logging loss and metric values
         S_data: list of array of joint visibility features for validation sequences {#seq}(#person x F x model.jsize/2)
         offset_miss: list of array of offset masks for validation sequences {#seq}(#person x F-1 x model.jsize)
         pose_miss: list of array of pose masks for validation sequences {#seq}(#person x F x model.jsize)
         scaler_pose: pose Scaler containing statistical information for normalization
         optimizer_g: optimizer used for training
         injection_step: current step in curriculum learning
    :return: val_loss: current validation loss
         T_val_loss: list of current and previous step validation losses
         Val_metricVal: list of metric value of current and previous last time step
         All_val: list of metric value of current and previous all time step
    """
    print("Validating model ...")
    forward_only = True
    val_step = 1
    val_loss = 0

    globalp_vel, globalt, globalp_occ= [], [], []
    # while val_step * val_batch_size < val_set_size:
    val_step_loss = 0
    #for each video sequence
    for seq in range(len(offsetData_val)):
        # Evaluate the model on the validation batches

        encoder_inputs, decoder_outputs, gt_pose_vals, lastseen_poses_val, Mask_offset, Mask_pose, Mask_pose_obs = model.get_batch_test(
                offsetData_val, seq, PoseDatavalidation, cfg, S_data, offset_miss, pose_miss, injection_step)
        encoder_inputs = np.transpose(encoder_inputs, (1, 0, 2))
        decoder_outputs = np.transpose(decoder_outputs, (1, 0, 2))

        I3d_feature = np.zeros((1, cfg.dataset.context_dim), dtype=float)
        if cfg.trainmode.use_context and cfg.dataset.dataset_name == 'posetrack':
            I3d_feature[0,:] = np.load(cfg.dataset.I3d_features_path + '/' + validation_set_fileNames[seq].split('.json')[0] + '/joint/i3dfeature.npy', allow_pickle=True)
        if cfg.trainmode.use_context and cfg.dataset.dataset_name == '3dpw':
            split_idx = np.load(cfg.dataset.split_data_path + '/split_idx_validation.pkl', allow_pickle=True)
            I3d_feature[0,:] = np.load(cfg.dataset.I3d_features_path + '/' + validation_set_fileNames[seq] + '_' + str(split_idx[seq]) + '/joint/i3dfeature.npy', allow_pickle=True)
        I3d_feature = send_to(num2tensor(I3d_feature), settings.device)

        obj_features = []
        if cfg.trainmode.human2obj:
            if cfg.dataset.dataset_name == '3dpw':
                fname = validation_set_fileNames[seq] + '_' + str(split_idx[seq])
                seq_name = validation_set_fileNames[seq]
            else:
                fname = validation_set_fileNames[seq].split('.json')[0]
                seq_name = validation_set_fileNames[seq].split('.json')[0]
            object = load_object('./' + cfg.dataset.obj_features_path + '/' + fname + '/detections.pkl')
            obj_features=get_object_features(object, cfg, seq_name)
        obj_features = np.array(obj_features)
        obj_features = send_to(num2tensor(obj_features), settings.device)

        model.eval()
        losses, obs_pred, val_poses = seq2seq_posetrack.init_step_curric(send_to(num2tensor(encoder_inputs), settings.device), send_to(num2tensor(decoder_outputs), settings.device), cfg, model,
            optimizer_g, True, send_to(num2tensor(Mask_offset), settings.device), send_to(num2tensor(Mask_pose), settings.device), injection_step, I3d_feature, obj_features)

        val_step_loss += sorted(losses.items())[0][1]

        val_loss += sorted(losses.items())[0][1]  # Loss book-keeping
        val_step += 1

        if cfg.trainmode.Add_visib == True and cfg.trainmode.Add_offset == True and cfg.trainmode.Add_pose == True:
            # val_pose_pred = convert2Pose(val_pred, lastseen_poses_val)   #***********
            pred_visib = val_poses[:, :, model.jsize * 2:].cpu().detach().numpy().transpose([1,0,2])
            val_poses = val_poses[:,:,model.jsize:model.jsize*2]
        elif cfg.trainmode.Add_visib == False and cfg.trainmode.Add_offset == True and cfg.trainmode.Add_pose == True:
            val_poses = val_poses[:,:,model.jsize:model.jsize*2]
        elif cfg.trainmode.Add_visib == False and cfg.trainmode.Add_offset == False and cfg.trainmode.Add_pose == True:
            val_poses = val_poses

        val_pose_pred = scaler_pose.inverse_transform(val_poses.cpu().detach().numpy().reshape(-1,val_poses.shape[2])).reshape(-1,val_poses.shape[1],val_poses.shape[2]).transpose([1,0,2])
        #scaler_pose.inverse_transform(val_poses.cpu().detach().numpy()).transpose([1,0,2])

        decoder_outputs = np.transpose(decoder_outputs, (1, 0, 2))
        if cfg.trainmode.Add_visib == True and cfg.trainmode.Add_offset == True and cfg.trainmode.Add_pose == True:
            decoder_outputs = decoder_outputs[:,:, model.jsize:model.jsize*2]
        elif cfg.trainmode.Add_visib == False and cfg.trainmode.Add_offset == True and cfg.trainmode.Add_pose == True:
            decoder_outputs = decoder_outputs[:,:, model.jsize:model.jsize*2]
        elif cfg.trainmode.Add_visib == False and cfg.trainmode.Add_offset == False and cfg.trainmode.Add_pose == True:
            decoder_outputs = decoder_outputs

        decoder_outputs = scaler_pose.inverse_transform(decoder_outputs.reshape(-1,decoder_outputs.shape[2])).reshape(-1,decoder_outputs.shape[1],decoder_outputs.shape[2])
        #scaler_pose.inverse_transform(decoder_outputs)



        for person in range(decoder_outputs.shape[0]):
            gts = decoder_outputs[person, 0:injection_step, :]

            # print("-----pose based ERROR:------")
            aa = Errors.GetErrors_PoseBased_withmask(val_pose_pred[person], gts, cfg, Mask_pose[person, 0:injection_step, :])
            globalp_vel.append(aa)
            if cfg.dataset.dataset_name == "posetrack":
                if cfg.trainmode.Add_visib == True:
                    pred_visib = np.where(pred_visib < 0, 0, 1)
                else:
                    pred_visib = np.zeros((decoder_outputs.shape[0], decoder_outputs.shape[1], int(model.jsize/2))) + 1

                aa_occ = occ_met(val_pose_pred[person], gts, cfg, pred_visib[person])
                globalp_occ.append(aa_occ)

    val_step_loss = val_step_loss / len(offsetData_val)
    writer.add_scalar('loss/lossValidation', val_step_loss, current_step)
    Avgglobalp = np.mean(globalp_vel, axis=0)

    writer.add_scalar('MetricValidation/err80_summary', Avgglobalp[1] * cfg.dataset.W_Scale,current_step) if injection_step >= 2 else None
    writer.add_scalar('MetricValidation/err160_summary', Avgglobalp[3] * cfg.dataset.W_Scale,current_step) if injection_step >= 4 else None
    writer.add_scalar('MetricValidation/err320_summary', Avgglobalp[7] * cfg.dataset.W_Scale,current_step) if injection_step >= 8 else None
    writer.add_scalar('MetricValidation/err400_summary', Avgglobalp[9] * cfg.dataset.W_Scale,current_step) if injection_step >= 10 else None
    writer.add_scalar('MetricValidation/err560_summary', Avgglobalp[13] * cfg.dataset.W_Scale,current_step) if injection_step >= 14 else None
    if cfg.dataset.dataset_name == "posetrack":
        Avgglobalp_occ = np.mean(globalp_occ, axis=0)
        writer.add_scalar('MetricValidation_occ/err80_summary', Avgglobalp_occ[1] * cfg.dataset.W_Scale, current_step) if injection_step >= 2 else None
        writer.add_scalar('MetricValidation_occ/err160_summary', Avgglobalp_occ[3] * cfg.dataset.W_Scale, current_step) if injection_step >= 4 else None
        writer.add_scalar('MetricValidation_occ/err320_summary', Avgglobalp_occ[7] * cfg.dataset.W_Scale, current_step) if injection_step >= 8 else None
        writer.add_scalar('MetricValidation_occ/err400_summary', Avgglobalp_occ[9] * cfg.dataset.W_Scale, current_step) if injection_step >= 10 else None
        writer.add_scalar('MetricValidation_occ/err560_summary', Avgglobalp_occ[13] * cfg.dataset.W_Scale, current_step) if injection_step >= 14 else None

    with LoggingPrinter(settings.log_file):
        print("")
        print("-------------------------------Validation Metric-------------------------------")
        print("{0: <16} |".format("milliseconds"), end="")
        for ms in [80, 160, 320, 400, 560, 1000]:
            print(" {0:6d} |".format(ms), end="")
        print()
        print("Ignored_occ{0: <5} |".format(""), end="")

        # print("~~~~~~~~Validation Metric~~~~~~~~~~")
        # print("Global Metric_validation:")
        tmprun = []
        max_ms = 1
        for ms in [1, 3, 7, 9, 13, 24]:
            if injection_step >= ms + 1:
                print(" {0:.3f} |".format(Avgglobalp[ms]* cfg.dataset.W_Scale), end="")
                tmprun.append(Avgglobalp[ms])
                if max_ms < ms:
                    max_ms = ms
            else:
                print("   n/a |", end="")
        print()
    Val_metricVal.append(Avgglobalp[max_ms])

    # if current_step <= 10000:
    All_val.append(tmprun)

    val_loss = val_loss / val_step
    T_val_loss.append(val_loss)
    if cfg.dataset.dataset_name == "posetrack":
        with LoggingPrinter(settings.log_file):
            print("consider_occ{0: <4} |".format(""), end="")
            for ms in [1, 3, 7, 9, 13, 24]:
                if injection_step >= ms + 1:
                    print(" {0:.3f} |".format(Avgglobalp_occ[ms]* cfg.dataset.W_Scale), end="")
                else:
                    print("   n/a |", end="")
            print()

    return val_loss, T_val_loss, Val_metricVal, All_val


def computeMetricForTestorTrain_curric(cfg, model, offset_test, PoseDatatest, metricVal, All_test, mode,
                                test_set_fileNames, current_step, writer, is_sample, S_data, offset_miss, pose_miss, scaler_pose, optimizer_g, injection_step):
    """ forward test/train data seq by seq and compute metric
        :param model:trained seq2seq model
             offset_test: list of array of offset information for test/train sequences {#seq}(#person x F-1 x model.jsize)
             PoseDatatest: list of array of pose locations for test/train sequences {#seq}(#person x F x model.jsize)
             metricVal: list of metric value of previous last time step
             All_test: list of metric value of previous all time step
             mode: test/train
             test_set_fileNames: file names of test/train split
             current_step: total current step of training
             writer: the writer used for logging loss and metric values
             is_sample:
             S_data: list of array of joint visibility features for test/train sequences {#seq}(#person x F x model.jsize/2)
             offset_miss: list of array of offset masks for test/train sequences {#seq}(#person x F-1 x model.jsize)
             pose_miss: list of array of pose masks for test/train sequences {#seq}(#person x F x model.jsize)
             scaler_pose: pose Scaler containing statistical information for normalization
             optimizer_g: optimizer used for training
             injection_step: current step in curriculum learning
        :return: metricVal: list of metric value of current and previous last time step
             All_test: list of metric value of current and previous all time step
        """
    if mode == "test":
        print("Testing model on test set ...")
    else:
        print("Testing model on train set ...")


    globalp = []
    globalt = []

    ToTal_test_loss = 0

    for seq in range(len(offset_test)):
        # Evaluate the model on the test batches
        encoder_inputs, decoder_outputs, gt_pose_vals, lastseen_poses_val, Mask_offset, Mask_pose, Mask_pose_obs = model.get_batch_test(
                offset_test, seq, PoseDatatest, cfg, S_data, offset_miss, pose_miss, injection_step)

        encoder_inputs = np.transpose(encoder_inputs, (1, 0, 2))
        decoder_outputs = np.transpose(decoder_outputs, (1, 0, 2))

        I3d_feature = np.zeros((1, 1024), dtype=float)
        if cfg.trainmode.use_context and cfg.dataset.dataset_name == 'posetrack':
            I3d_feature[0,:] = np.load(cfg.dataset.I3d_features_path + '/' + test_set_fileNames[seq].split('.json')[0] + '/joint/i3dfeature.npy', allow_pickle=True)
        if cfg.trainmode.use_context and cfg.dataset.dataset_name == '3dpw':
            split_idx = np.load(cfg.dataset.split_data_path + '/split_idx_test.pkl', allow_pickle=True)
            I3d_feature[0,:] = np.load(cfg.dataset.I3d_features_path + '/' + test_set_fileNames[seq] + '_' + str(split_idx[seq]) + '/joint/i3dfeature.npy', allow_pickle=True)
        I3d_feature = send_to(num2tensor(I3d_feature), settings.device)

        obj_features = []
        if cfg.trainmode.human2obj:
            if cfg.dataset.dataset_name == '3dpw':
                fname = test_set_fileNames[seq] + '_' + str(split_idx[seq])
                seq_name = test_set_fileNames[seq]
            else:
                fname = test_set_fileNames[seq].split('.json')[0]
                seq_name = test_set_fileNames[seq].split('.json')[0]
            object = load_object('./' + cfg.dataset.obj_features_path + '/' + fname + '/detections.pkl')
            obj_features=get_object_features(object, cfg, seq_name)
        obj_features = np.array(obj_features)
        obj_features = send_to(num2tensor(obj_features), settings.device)


        model.eval()
        losses, obs_pred, test_poses = seq2seq_posetrack.init_step_curric(send_to(num2tensor(encoder_inputs), settings.device), send_to(num2tensor(decoder_outputs), settings.device), cfg, model,
                optimizer_g, True, send_to(num2tensor(Mask_offset), settings.device), send_to(num2tensor(Mask_pose), settings.device), injection_step, I3d_feature, obj_features)

        ToTal_test_loss += sorted(losses.items())[0][1]


        # Select pose part from input features
        if cfg.trainmode.Add_visib == True and cfg.trainmode.Add_offset == True and cfg.trainmode.Add_pose == True:
            # test_pose_pred = convert2Pose(test_pred, lastseen_poses_val)   #***********
            pred_visib = test_poses[:, :, model.jsize * 2:].cpu().detach().numpy().transpose([1, 0, 2])
            test_poses = test_poses[:,:,model.jsize:model.jsize*2]
        elif cfg.trainmode.Add_visib == False and cfg.trainmode.Add_offset == True and cfg.trainmode.Add_pose == True:
            test_poses = test_poses[:,:,model.jsize:model.jsize*2]
        elif cfg.trainmode.Add_visib == False and cfg.trainmode.Add_offset == False and cfg.trainmode.Add_pose == True:
            test_poses = test_poses


        # Denormalize the output
        test_pose_pred = scaler_pose.inverse_transform(test_poses.cpu().detach().numpy().reshape(-1,test_poses.shape[2])).reshape(-1,test_poses.shape[1],test_poses.shape[2]).transpose([1,0,2])
        #scaler_pose.inverse_transform(test_poses.cpu().detach().numpy()).transpose([1,0,2])


        # -----------------------------------------------------------#
        decoder_outputs = np.transpose(decoder_outputs, (1, 0, 2))
        encoder_inputs = np.transpose(encoder_inputs, (1, 0, 2))

        if cfg.trainmode.Add_visib == True and cfg.trainmode.Add_offset == True and cfg.trainmode.Add_pose == True:
            decoder_outputs = decoder_outputs[:,:, model.jsize:model.jsize*2]
            observation = encoder_inputs[:,:, model.jsize:model.jsize*2]
            # observation_end = decoder_inputs[:,0, int(decoder_inputs.shape[2] / 5) * 2:int(decoder_inputs.shape[2] / 5) * 4]
            # observation = np.concatenate((observation, np.expand_dims(observation_end, axis=1)), axis=1)
        elif cfg.trainmode.Add_visib == False and cfg.trainmode.Add_offset == True and cfg.trainmode.Add_pose == True:
            decoder_outputs = decoder_outputs[:,:, model.jsize:model.jsize*2]
            observation = encoder_inputs[:, :,model.jsize:model.jsize*2]
            # observation_end = decoder_inputs[:, 0,int(decoder_inputs.shape[2] / 4) * 2:int(decoder_inputs.shape[2] / 4) * 4]
            # observation = np.concatenate((observation, np.expand_dims(observation_end, axis=1)), axis=1)
        elif cfg.trainmode.Add_visib == False and cfg.trainmode.Add_offset == False and cfg.trainmode.Add_pose == True:
            decoder_outputs = decoder_outputs
            observation = encoder_inputs
            # observation_end = decoder_inputs[:,0,:]

        observation = scaler_pose.inverse_transform(observation.reshape(-1,observation.shape[2])).reshape(-1,observation.shape[1],observation.shape[2])
        decoder_outputs = scaler_pose.inverse_transform(decoder_outputs.reshape(-1,decoder_outputs.shape[2])).reshape(-1,decoder_outputs.shape[1],decoder_outputs.shape[2])

        for person in range(decoder_outputs.shape[0]):
            gts = decoder_outputs[person, 0:injection_step, :]

            aa = Errors.GetErrors_PoseBased_withmask(test_pose_pred[person], gts, cfg, Mask_pose[person, 0:injection_step, :])
            globalp.append(aa)
            if cfg.dataset.dataset_name == "posetrack":
                if cfg.trainmode.Add_visib == True:
                    pred_visib = np.where(pred_visib < 0, 0, 1)
                else:
                    pred_visib = np.zeros(
                        (decoder_outputs.shape[0], decoder_outputs.shape[1], int(model.jsize / 2))) + 1

    ToTal_test_loss = ToTal_test_loss / len(offset_test)
    if not is_sample and mode == 'test':
        writer.add_scalar('loss/lossTest', ToTal_test_loss, current_step)
    if not is_sample and mode == 'train':
        writer.add_scalar('loss/lossTraaaaaaain', ToTal_test_loss, current_step)

    Avgglobalp = np.mean(globalp, axis=0)

    if mode == 'test' and is_sample==0:
        with LoggingPrinter(settings.log_file):
            print("-------------------------------test Metric---------------------------------")
        # print("Global All person Average Error_test:")
        writer.add_scalar('MetricTest/err80_summary', Avgglobalp[1]* cfg.dataset.W_Scale,current_step) if injection_step >= 2 else None
        writer.add_scalar('MetricTest/err160_summary', Avgglobalp[3]* cfg.dataset.W_Scale,current_step) if injection_step >= 4 else None
        writer.add_scalar('MetricTest/err320_summary', Avgglobalp[7]* cfg.dataset.W_Scale,current_step) if injection_step >= 8 else None
        writer.add_scalar('MetricTest/err400_summary', Avgglobalp[9]* cfg.dataset.W_Scale,current_step) if injection_step >= 10 else None
        writer.add_scalar('MetricTest/err560_summary', Avgglobalp[13]* cfg.dataset.W_Scale,current_step) if injection_step >= 14 else None



    elif mode=='train':
        with LoggingPrinter(settings.log_file):
            print("-------------------------------train Metric--------------------------------")
        writer.add_scalar('MetricTrain/err80_summary', Avgglobalp[1] * cfg.dataset.W_Scale,current_step) if injection_step >= 2 else None
        writer.add_scalar('MetricTrain/err160_summary', Avgglobalp[3] * cfg.dataset.W_Scale,current_step) if injection_step >= 4 else None
        writer.add_scalar('MetricTrain/err320_summary', Avgglobalp[7] * cfg.dataset.W_Scale,current_step) if injection_step >= 8 else None
        writer.add_scalar('MetricTrain/err400_summary', Avgglobalp[9] * cfg.dataset.W_Scale, current_step) if injection_step >= 10 else None
        writer.add_scalar('MetricTrain/err560_summary', Avgglobalp[13] * cfg.dataset.W_Scale,current_step) if injection_step >= 14 else None


    with LoggingPrinter(settings.log_file):
        print("{0: <16} |".format("milliseconds"), end="")
        for ms in [80, 160, 320, 400, 560, 1000]:
            print(" {0:5d} |".format(ms), end="")
        print()
        print("Ignored_occ{0: <5} |".format(""), end="")

        tmprun = []
        max_ms = 1
        for ms in [1, 3, 7, 9, 13, 24]:
            if injection_step >= ms + 1:
                print(" {0:.3f} |".format(Avgglobalp[ms]* cfg.dataset.W_Scale), end="")
                tmprun.append(Avgglobalp[ms])
                if max_ms < ms:
                    max_ms = ms
            else:
                print("   n/a |", end="")
        print()
    metricVal.append(Avgglobalp[max_ms])

    # if current_step <= 10000:
    All_test.append(tmprun)


    with LoggingPrinter(settings.log_file):
        print("Center pose{0: <4} |".format(""), end="")
        for ms in [1, 3, 7, 9, 13, 24]:
            if injection_step >= ms + 1:
                print(" {0:.3f} |".format(Avgglobalssp[ms] * cfg.dataset.W_Scale), end="")
            else:
                print("   n/a |", end="")
        print()
    with LoggingPrinter(settings.log_file):
        print("trajectory{0: <4} |".format(""), end="")
        for ms in [1, 3, 7, 9, 13, 24]:
            if injection_step >= ms + 1:
                print(" {0:.3f} |".format(Avgglobalssn[ms] * cfg.dataset.W_Scale), end="")
            else:
                print("   n/a |", end="")
        print()

    return metricVal, All_test

