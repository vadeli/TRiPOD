import random
import os
import logging
import sys
import numpy as np
from six.moves import xrange
import gc
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utility.read_data import read_all_data, read_all_data_3DPW
from utility import utils
from models import seq2seq
from models.losses import get_total_norm
from utility.classes import LoggingPrinter
from utility.utils import num2tensor, send_to
import forward
import utility.settings as settings

from utility.data_utils_posetrack import create_seq_dimensions


FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def Curriculum_social_PT(cfg, summaries_dir):
    """Train a seq2seq model on human poses"""
    test_flag = 1
    # Setup parameters
    train_dir = summaries_dir.split('summary')[0] + 'checkpoints' + summaries_dir.split('summary')[1]
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    settings.init(summaries_dir.split('summary')[0]+'logs/logs_'+summaries_dir.split('summary/')[1]+'.txt')
    # log_file = summaries_dir.split('summary')[0]+'logs/logs_'+summaries_dir.split('summary/')[1]+'.txt'

    cfg.defrost()
    cfg.trainmode.centered = 0
    cfg.trainmode.traj = 0
    # if FLAGS.Add_offset == 1:
    cfg.model.seq_length_in = cfg.model.seq_length_in - 1
    cfg.freeze()

    offsetData, poseData, shapedata, fileNames, visibData, dummy, miss_masks, scaler_offset, scaler_pose = read_all_data(cfg, 1)
    # create_seq_dimensions(fileNames)
    if cfg.learning.shuffle == True:
        #shuffle offset and pose locations
        c = list(zip(offsetData.train, poseData.train))
        random.shuffle(c)
        offsetData.train, poseData.train = zip(*c)

    offset_AllPerson = {
        'train': [],
        'test': [],
        'validation': []
    }
    pose_AllPerson = {
        'train': [],
        'test': [],
        'validation': []
    }

    for t in range(len(offsetData['train'])):
        for p in range(offsetData['train'][t].shape[0]):
            offset_AllPerson['train'].append(offsetData['train'][t][p, :, :])

    for t in range(len(offsetData['test'])):
        for p in range(offsetData['test'][t].shape[0]):
            offset_AllPerson['test'].append(offsetData['test'][t][p, :, :])

    for t in range(len(offsetData['validation'])):
        for p in range(offsetData['validation'][t].shape[0]):
            offset_AllPerson['validation'].append(offsetData['validation'][t][p, :, :])

    for t in range(len(poseData['validation'])):
        for p in range(poseData['validation'][t].shape[0]):
            pose_AllPerson['validation'].append(poseData['validation'][t][p, :, :])

    for t in range(len(poseData['train'])):
        for p in range(poseData['train'][t].shape[0]):
            pose_AllPerson['train'].append(poseData['train'][t][p, :, :])

    S_Datatrain_AllPerson = []
    for t in range(len(visibData['train'])):
        for p in range(visibData['train'][t].shape[0]):
            S_Datatrain_AllPerson.append(visibData['train'][t][p, :, :])
    # for p in range(S_data_train.shape[0]):
    #     S_Datatrain_AllPerson.append(S_data_train[p, :, :])

    MissMaskoffset_train_AllPerson = []
    for t in range(len(miss_masks.offset_train)):
        for p in range(miss_masks.offset_train[t].shape[0]):
            MissMaskoffset_train_AllPerson.append(miss_masks.offset_train[t][p, :, :])

    MissMaskpose_train_AllPerson = []
    for t in range(len(miss_masks.pose_train)):
        for p in range(miss_masks.pose_train[t].shape[0]):
            MissMaskpose_train_AllPerson.append(miss_masks.pose_train[t][p, :, :])

    print(torch.cuda.get_device_name(0))

    long_dtype, float_dtype = utils.get_dtypes(cfg)

    # === Create the model ===
    model, checkpoint, optimizer_g, current_step, epoch, m_ep, m_st = init_model(shapedata, float_dtype, cfg)
    model.cuda()   #or model.to(device)

    # print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    # current_step = 0 if FLAGS.load <= 0 else model.global_step.eval()

    running_loss = 0.0
    T_val_loss, Val_metricVal, All_val = [], [], []
    metricValTest, All_test = [], []
    metricValTrain, All_train = [], []

    Step_num = cfg.model.seq_length_out / cfg.model.injectLayer_num


    writer = SummaryWriter(os.path.normpath(os.path.join(summaries_dir)))

    # writer.add_graph(model, torch.from_numpy(encoder_inputs).cuda().to(torch.float32), 2, model.seq_start_end)
    for stp in range(epoch, int(Step_num)):
        injection_step = (stp * cfg.model.injectLayer_num) + cfg.model.injectLayer_num

        with LoggingPrinter(settings.log_file):
            print("********************step: {}*********************".format(injection_step))
        for ep in range(m_ep, cfg.learning.epoch_num):
            for st in xrange(m_st, int(np.ceil(len(offsetData['train']) / cfg.learning.batch_size))):

                forward_only = False
                gc.collect()

                encoder_inputs, decoder_outputs, gt_pose_vals, lastseen_poses_val, Mask_offset, Mask_pose, I3d_feature, obj_features = model.get_batch(
                    offsetData['train'], st, poseData['train'], cfg, visibData['train'], fileNames['train'], miss_masks.offset_train, miss_masks.pose_train, injection_step)
                encoder_inputs = np.transpose(encoder_inputs, (1, 0, 2))
                decoder_outputs = np.transpose(decoder_outputs, (1, 0, 2))

                encoder_inputs = send_to(num2tensor(encoder_inputs), settings.device)
                decoder_outputs = send_to(num2tensor(decoder_outputs), settings.device)
                Mask_offset = send_to(num2tensor(Mask_offset), settings.device)
                Mask_pose = send_to(num2tensor(Mask_pose), settings.device)
                I3d_feature = send_to(num2tensor(I3d_feature), settings.device)
                obj_features = send_to(num2tensor(obj_features), settings.device)

                if test_flag == 0:
                    with torch.set_grad_enabled(True):
                        model.train()
                        losses, obs_pred, pred = seq2seq.init_step_curric(encoder_inputs, decoder_outputs, cfg, model,
                                                                                    optimizer_g, forward_only, Mask_offset, Mask_pose, injection_step, I3d_feature, obj_features)
                    running_loss += sorted(losses.items())[0][1]

                    if current_step % cfg.experiment.test_every == 0:  # every 1000 mini-batches...
                        if current_step==0:
                            writer.add_scalar('Loss/train', running_loss, current_step)
                            with LoggingPrinter(settings.log_file):
                                print("epoch {}/{}; step {}; step_loss: {}".format(ep, cfg.learning.epoch_num, current_step, running_loss))
                        else:
                            writer.add_scalar('Loss/train', running_loss / cfg.experiment.test_every, current_step)
                            with LoggingPrinter(settings.log_file):
                                print()
                                print("epoch {}/{}; step {}; step_loss: {}".format(ep, cfg.learning.epoch_num, current_step,
                                                                                   running_loss / cfg.experiment.test_every))
                        #create historgram of weights
                        for name, param in model.named_parameters():
                            if param.requires_grad:
                                # print(name)#, param.data)
                                writer.add_histogram(name, param.data)

                        for k, v in sorted(losses.items()):
                            # logger.info('  [G] {}: {:.3f}'.format(k, v))
                            checkpoint['losses'].append(v)
                        checkpoint['sample_ts'].append(current_step)
                        checkpoint['counters']['current_step'] = current_step
                        checkpoint['counters']['epoch'] = stp
                        checkpoint['counters']['m_ep'] = ep
                        checkpoint['counters']['m_st'] = st

                        running_loss = 0.0

                        with torch.set_grad_enabled(False):
                            #validating the model and computing metric for validation
                            val_loss, T_val_loss, Val_metricVal, All_val = forward.computeMetricForValidation_curric(cfg, model, offsetData['validation'],
                                                                                                      poseData['validation'],
                                                                                                      Val_metricVal, All_val, T_val_loss,
                                                                                                      fileNames['validation'],
                                                                                                      current_step,writer,visibData['validation'],
                                                                                                      miss_masks.offset_validation,
                                                                                                      miss_masks.pose_validation,
                                                                                                      scaler_pose, optimizer_g, injection_step)


                            metricValTest, All_test = forward.computeMetricForTestorTrain_curric(cfg, model, offsetData['test'], poseData['test'],
                                                                                  metricValTest, All_test, 'test', fileNames['test'], current_step,
                                                                                  writer, 0, visibData['test'], miss_masks.offset_test,
                                                                                  miss_masks.pose_test,scaler_pose, optimizer_g, injection_step)

                            # metricValTrain, All_train = forward.computeMetricForTestorTrain_curric(cfg, model, offsetData['train'], poseData['train'],
                            #                                                         metricValTrain, All_train, 'train', fileNames['train'], current_step,
                            #                                                         writer, 0, visibData['train'], miss_masks.offset_train,
                            #                                                         miss_masks.pose_train, scaler_pose, optimizer_g, injection_step)

                        checkpoint['model_state'] = model.state_dict()
                        checkpoint['model_optim_state'] = optimizer_g.state_dict()
                        checkpoint['model_struct'] = model

                        max_ms = 0
                        for ms in [1, 3, 7, 9, 13, 24]:
                            if injection_step >= ms + 1:
                                if max_ms < ms:
                                    max_ms = ms
                        last = [1, 3, 7, 9, 13, 24].index(max_ms)

                        if not os.path.exists(os.path.join(train_dir, 'best', str(injection_step))):
                            os.makedirs(os.path.join(train_dir, 'best', str(injection_step)))
                        try:
                            best_mertic_val = np.load(os.path.join(train_dir, 'best', str(injection_step), 'best_mertic_val.npy'))
                        except:
                            try:
                                best_mertic_val = All_val[-1][last]
                            except:
                                ppp=1
                        if best_mertic_val >= All_val[-1][last]:
                            best_mertic_val = All_val[-1][last]
                            iter_val = current_step

                            for filename in os.listdir(os.path.join(train_dir, 'best', str(injection_step))):
                                file_path = os.path.join(os.path.join(train_dir, 'best', str(injection_step)), filename)
                                os.unlink(file_path)

                            print("Saving the best model on validation...");
                            np.save(os.path.join(train_dir, 'best', str(injection_step),'best_mertic_val.npy'), best_mertic_val)
                            # model.saver.save(sess, os.path.normpath(os.path.join(train_dir, 'best',str(injection_step), 'best_checkpoint')),
                            #                  global_step=current_step)
                            checkpoint_path = os.path.join(train_dir, 'best', str(injection_step), 'best_checkpoint_torch_' + str(current_step))
                            for fl in os.listdir(os.path.join(train_dir, 'best_copy', str(injection_step))):
                                os.remove(os.path.join(train_dir, 'best_copy', str(injection_step),fl))
                            if not os.path.exists(os.path.join(train_dir, 'best_copy', str(injection_step))):
                                os.makedirs(os.path.join(train_dir, 'best_copy', str(injection_step)))
                            torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=False)
                            shutil.copy(checkpoint_path, os.path.join(train_dir, 'best_copy', str(injection_step),
                                                                      'best_checkpoint_torch_' + str(current_step)))
                        # _______________________________________________________________________________________
                        if not os.path.exists(os.path.join(train_dir, 'best_test', str(injection_step))):
                            os.makedirs(os.path.join(train_dir, 'best_test', str(injection_step)))
                        try:
                            best_mertic_test = np.load(os.path.join(train_dir, 'best_test',str(injection_step), 'best_mertic_test.npy'))
                        except:
                            best_mertic_test = All_test[-1][last]
                        if best_mertic_test >= All_test[-1][last]:
                            best_mertic_test = All_test[-1][last]
                            iter_test = current_step

                            for filename in os.listdir(os.path.join(train_dir, 'best_test',str(injection_step))):
                                file_path = os.path.join(os.path.join(train_dir, 'best_test',str(injection_step)), filename)
                                os.unlink(file_path)

                            print("Saving the best model on test...");
                            np.save(os.path.join(train_dir, 'best_test', str(injection_step),'best_mertic_test.npy'), best_mertic_test)
                            # model.saver.save(sess, os.path.normpath(os.path.join(train_dir, 'best_test', str(injection_step),'best_checkpoint')),
                            #                  global_step=current_step)
                            checkpoint_path = os.path.join(train_dir, 'best_test', str(injection_step),'best_checkpoint_torch_' + str(current_step))
                            torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=False)
                        # _______________________________________________________________________________________
                        if not os.path.exists(os.path.join(train_dir, 'best_avg_test', str(injection_step))):
                            os.makedirs(os.path.join(train_dir, 'best_avg_test', str(injection_step)))
                        try:
                            best_mertic_test_avg = np.load(
                                os.path.join(train_dir, 'best_avg_test', str(injection_step), 'best_mertic_test.npy'))
                        except:
                            try:
                                best_mertic_test_avg = np.mean(All_test[-1])
                            except:
                                best_mertic_test_avg = np.mean(All_test[-1])
                        if best_mertic_test_avg >= np.mean(All_test[-1]):
                            best_mertic_test_avg = np.mean(All_test[-1])
                            iter_test_avg = current_step

                            for filename in os.listdir(os.path.join(train_dir, 'best_avg_test', str(injection_step))):
                                file_path = os.path.join(os.path.join(train_dir, 'best_avg_test', str(injection_step)),
                                                         filename)
                                os.unlink(file_path)

                            print("Saving the best model on test AVG...");
                            np.save(
                                os.path.join(train_dir, 'best_avg_test', str(injection_step), 'best_mertic_test_avg.npy'),
                                best_mertic_test_avg)
                            # model.saver.save(sess, os.path.normpath(os.path.join(train_dir, 'best',str(injection_step), 'best_checkpoint')),
                            #                  global_step=current_step)
                            checkpoint_path = os.path.join(train_dir, 'best_avg_test', str(injection_step),
                                                           'best_checkpoint_torch_' + str(current_step))
                            torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=False)
                            print('Done Saving')
                            print("Training ...")
                            # torch.save(model.state_dict(), os.path.normpath(os.path.join(train_dir, 'best_test', str(injection_step),'best_checkpoint_torch')))
                        try:
                            f = open(summaries_dir.split('summary')[0] + 'logs/logs_' + summaries_dir.split('summary/')[
                                1] + '_bestIndex.txt', 'wt')
                            f.write("best validation: {}, best_test:{}, best_test_avg:{}".format(iter_val, iter_test, iter_test_avg))
                            f.close()
                        except:
                            ppp = 1

                        # writer.add_scalar('Loss/train', sorted(losses.items())[0][1], t)
                    # loss_summary = tf.summary.scalar('loss/loss', tf.convert_to_tensor(sorted(losses.items())[0][1]))
                    # model.train_writer.add_summary(loss_summary, t).eval()

                    checkpoint['norm_g'].append(
                        get_total_norm(model.parameters())
                    )

                    current_step += 1
                    if current_step >= cfg.learning.iterations:
                        break
                else:
                    metricValTest, All_test = forward.computeMetricForTestorTrain_curric(cfg, model, offsetData['test'],
                                                                                         poseData['test'],
                                                                                         metricValTest, All_test,
                                                                                         'test', fileNames['test'],
                                                                                         current_step,
                                                                                         writer, 0, visibData['test'],
                                                                                         miss_masks.offset_test,
                                                                                         miss_masks.pose_test,
                                                                                         scaler_pose, optimizer_g,
                                                                                         injection_step)

        # Load best previous model
        # checkpoint_path = os.path.join(train_dir, 'best_test', str(injection_step))
        checkpoint_path = os.path.join(train_dir, 'best_test', str(injection_step))
        for f in os.listdir(checkpoint_path):
            if f[0:15] == 'best_checkpoint':
                break

        checkpoint = torch.load(checkpoint_path+'/'+f)
        model.load_state_dict(checkpoint['model_state'])
        optimizer_g.load_state_dict(checkpoint['model_optim_state'])
        # current_step = checkpoint['counters']['current_step']
        # epoch = checkpoint['counters']['epoch']
        # checkpoint['sample_ts'].append(current_step)

        # torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=False)
        # latest = tf.train.latest_checkpoint(checkpoint_path)
        # model.saver.restore(sess, latest)
        print("Model loaded")
        #Visualize testset
        # Test_plot(model, offsetData['test'], poseData['test'], fileNames['test'], 'test', current_step, visibData['test'], miss_masks.offset_test,
        #         miss_masks.pose_test, scaler_pose, optimizer_g, injection_step)
    print('Process Ended!!!')

def Curriculum_social_3DPW(cfg, summaries_dir):
    """Train a seq2seq model on human poses"""

    # Setup parameters
    train_dir = summaries_dir.split('summary')[0] + 'checkpoints' + summaries_dir.split('summary')[1]
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    settings.init(summaries_dir.split('summary')[0]+'logs/logs_'+summaries_dir.split('summary/')[1]+'.txt')
    # log_file = summaries_dir.split('summary')[0]+'logs/logs_'+summaries_dir.split('summary/')[1]+'.txt'

    cfg.defrost()
    cfg.trainmode.centered = 0
    cfg.trainmode.traj = 0
    # if FLAGS.Add_offset == 1:
    cfg.model.seq_length_in = cfg.model.seq_length_in - 1
    cfg.freeze()

    offsetData, poseData, shapedata, fileNames, scaler_pose = read_all_data_3DPW(cfg, 1)

    if cfg.learning.shuffle == True:
        #shuffle offset and pose locations
        c = list(zip(offsetData.train, poseData.train))
        random.shuffle(c)
        offsetData.train, poseData.train = zip(*c)

    offset_AllPerson = {
        'train': [],
        'test': [],
        'validation': []
    }
    pose_AllPerson = {
        'train': [],
        'test': [],
        'validation': []
    }

    for t in range(len(offsetData['train'])):
        for p in range(offsetData['train'][t].shape[0]):
            offset_AllPerson['train'].append(offsetData['train'][t][p, :, :])

    for t in range(len(offsetData['test'])):
        for p in range(offsetData['test'][t].shape[0]):
            offset_AllPerson['test'].append(offsetData['test'][t][p, :, :])

    for t in range(len(offsetData['validation'])):
        for p in range(offsetData['validation'][t].shape[0]):
            offset_AllPerson['validation'].append(offsetData['validation'][t][p, :, :])

    for t in range(len(poseData['validation'])):
        for p in range(poseData['validation'][t].shape[0]):
            pose_AllPerson['validation'].append(poseData['validation'][t][p, :, :])

    for t in range(len(poseData['train'])):
        for p in range(poseData['train'][t].shape[0]):
            pose_AllPerson['train'].append(poseData['train'][t][p, :, :])


    print(torch.cuda.get_device_name(0))

    long_dtype, float_dtype = utils.get_dtypes(cfg)

    # === Create the model ===
    model, checkpoint, optimizer_g, current_step, epoch, m_ep, m_st = init_model(shapedata, float_dtype, cfg)
    model.cuda()   #or model.to(device)

    # print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    # current_step = 0 if FLAGS.load <= 0 else model.global_step.eval()

    running_loss = 0.0
    T_val_loss, Val_metricVal, All_val = [], [], []
    metricValTest, All_test = [], []
    metricValTrain, All_train = [], []

    Step_num = cfg.model.seq_length_out / cfg.model.injectLayer_num


    writer = SummaryWriter(os.path.normpath(os.path.join(summaries_dir)))

    # writer.add_graph(model, torch.from_numpy(encoder_inputs).cuda().to(torch.float32), 2, model.seq_start_end)
    for stp in range(epoch, int(Step_num)):
        injection_step = (stp * cfg.model.injectLayer_num) + cfg.model.injectLayer_num

        with LoggingPrinter(settings.log_file):
            print("********************step: {}*********************".format(injection_step))
        for ep in range(m_ep, cfg.learning.epoch_num):
            for st in xrange(m_st, int(np.ceil(len(offsetData['train']) / cfg.learning.batch_size))):

                forward_only = False
                gc.collect()

                encoder_inputs, decoder_outputs, gt_pose_vals, lastseen_poses_val, _, _, I3d_feature, obj_features = model.get_batch(
                    offsetData['train'], st, poseData['train'], cfg, [], fileNames['train'], [], [], injection_step)
                encoder_inputs = np.transpose(encoder_inputs, (1, 0, 2))
                decoder_outputs = np.transpose(decoder_outputs, (1, 0, 2))

                encoder_inputs = send_to(num2tensor(encoder_inputs), settings.device)
                decoder_outputs = send_to(num2tensor(decoder_outputs), settings.device)
                I3d_feature = send_to(num2tensor(I3d_feature), settings.device)
                obj_features = send_to(num2tensor(obj_features), settings.device)

                with torch.set_grad_enabled(True):
                    model.train()
                    losses, obs_pred, pred = seq2seq.init_step_curric(encoder_inputs, decoder_outputs, cfg, model,
                                                                                optimizer_g, forward_only, [], [], injection_step, I3d_feature, obj_features)
                running_loss += sorted(losses.items())[0][1]

                if current_step % cfg.experiment.test_every == 0:  # every 1000 mini-batches...
                    if current_step==0:
                        writer.add_scalar('Loss/train', running_loss, current_step)
                        with LoggingPrinter(settings.log_file):
                            print("epoch {}/{}; step {}; step_loss: {}".format(ep, cfg.learning.epoch_num, current_step, running_loss))
                    else:
                        writer.add_scalar('Loss/train', running_loss / cfg.experiment.test_every, current_step)
                        with LoggingPrinter(settings.log_file):
                            print()
                            print("epoch {}/{}; step {}; step_loss: {}".format(ep, cfg.learning.epoch_num, current_step,
                                                                               running_loss / cfg.experiment.test_every))
                    #create historgram of weights
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            # print(name)#, param.data)
                            writer.add_histogram(name, param.data)

                    for k, v in sorted(losses.items()):
                        # logger.info('  [G] {}: {:.3f}'.format(k, v))
                        checkpoint['losses'].append(v)
                    checkpoint['sample_ts'].append(current_step)
                    checkpoint['counters']['current_step'] = current_step
                    checkpoint['counters']['epoch'] = stp
                    checkpoint['counters']['m_ep'] = ep
                    checkpoint['counters']['m_st'] = st

                    running_loss = 0.0

                    with torch.set_grad_enabled(False):
                        #validating the model and computing metric for validation
                        val_loss, T_val_loss, Val_metricVal, All_val = forward.computeMetricForValidation_curric(cfg, model, offsetData['validation'],
                                                                                                  poseData['validation'],
                                                                                                  Val_metricVal, All_val, T_val_loss,
                                                                                                  fileNames['validation'],
                                                                                                  current_step,writer,
                                                                                                  [],[],[],
                                                                                                  scaler_pose, optimizer_g, injection_step)


                        metricValTest, All_test = forward.computeMetricForTestorTrain_curric(cfg, model, offsetData['test'], poseData['test'],
                                                                              metricValTest, All_test, 'test', fileNames['test'], current_step,
                                                                              writer, 0, [], [],[], scaler_pose, optimizer_g, injection_step)

                        # metricValTrain, All_train = forward.computeMetricForTestorTrain_curric(cfg, model, offsetData['train'], poseData['train'],
                        #                                                         metricValTrain, All_train, 'train', fileNames['train'], current_step,
                        #                                                         writer, 0, visibData['train'], miss_masks.offset_train,
                        #                                                         miss_masks.pose_train, scaler_pose, optimizer_g, injection_step)

                    checkpoint['model_state'] = model.state_dict()
                    checkpoint['model_optim_state'] = optimizer_g.state_dict()
                    checkpoint['model_struct'] = model

                    max_ms = 0
                    for ms in [1, 3, 7, 9, 13, 24]:
                        if injection_step >= ms + 1:
                            if max_ms < ms:
                                max_ms = ms
                    last = [1, 3, 7, 9, 13, 24].index(max_ms)

                    if not os.path.exists(os.path.join(train_dir, 'best', str(injection_step))):
                        os.makedirs(os.path.join(train_dir, 'best', str(injection_step)))
                    try:
                        best_mertic_val = np.load(os.path.join(train_dir, 'best', str(injection_step), 'best_mertic_val.npy'))
                    except:
                        try:
                            best_mertic_val = All_val[-1][last]
                        except:
                            ppp=1
                    if best_mertic_val >= All_val[-1][last]:
                        best_mertic_val = All_val[-1][last]
                        iter_val = current_step

                        for filename in os.listdir(os.path.join(train_dir, 'best', str(injection_step))):
                            file_path = os.path.join(os.path.join(train_dir, 'best', str(injection_step)), filename)
                            os.unlink(file_path)

                        print("Saving the best model on validation...");
                        np.save(os.path.join(train_dir, 'best', str(injection_step),'best_mertic_val.npy'), best_mertic_val)
                        # model.saver.save(sess, os.path.normpath(os.path.join(train_dir, 'best',str(injection_step), 'best_checkpoint')),
                        #                  global_step=current_step)
                        checkpoint_path = os.path.join(train_dir, 'best', str(injection_step), 'best_checkpoint_torch_' + str(current_step))
                        torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=False)

                    if not os.path.exists(os.path.join(train_dir, 'best_test', str(injection_step))):
                        os.makedirs(os.path.join(train_dir, 'best_test', str(injection_step)))
                    try:
                        best_mertic_test = np.load(os.path.join(train_dir, 'best_test',str(injection_step), 'best_mertic_test.npy'))
                    except:
                        best_mertic_test = All_test[-1][last]
                    if best_mertic_test >= All_test[-1][last]:
                        best_mertic_test = All_test[-1][last]
                        iter_test = current_step

                        for filename in os.listdir(os.path.join(train_dir, 'best_test',str(injection_step))):
                            file_path = os.path.join(os.path.join(train_dir, 'best_test',str(injection_step)), filename)
                            os.unlink(file_path)

                        print("Saving the best model on test...");
                        np.save(os.path.join(train_dir, 'best_test', str(injection_step),'best_mertic_test.npy'), best_mertic_test)
                        # model.saver.save(sess, os.path.normpath(os.path.join(train_dir, 'best_test', str(injection_step),'best_checkpoint')),
                        #                  global_step=current_step)
                        checkpoint_path = os.path.join(train_dir, 'best_test', str(injection_step),'best_checkpoint_torch_' + str(current_step))
                        torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=False)
                        print('Done Saving')
                        print("Training ...")
                        # torch.save(model.state_dict(), os.path.normpath(os.path.join(train_dir, 'best_test', str(injection_step),'best_checkpoint_torch')))

                    try:
                        f = open(summaries_dir.split('summary')[0] + 'logs/logs_' + summaries_dir.split('summary/')[
                            1] + '_bestIndex.txt', 'wt')
                        f.write("best validation: {}, best_test:{}".format(iter_val,iter_test))
                        f.close()
                    except:
                        ppp=1



                    # writer.add_scalar('Loss/train', sorted(losses.items())[0][1], t)
                # loss_summary = tf.summary.scalar('loss/loss', tf.convert_to_tensor(sorted(losses.items())[0][1]))
                # model.train_writer.add_summary(loss_summary, t).eval()

                checkpoint['norm_g'].append(
                    get_total_norm(model.parameters())
                )

                current_step += 1
                if current_step >= cfg.learning.iterations:
                    break
        # Load best previous model
        # checkpoint_path = os.path.join(train_dir, 'best_test', str(injection_step))
        checkpoint_path = os.path.join(train_dir, 'best_test', str(injection_step))
        for f in os.listdir(checkpoint_path):
            if f[0:15] == 'best_checkpoint':
                break

        checkpoint = torch.load(checkpoint_path+'/'+f)
        model.load_state_dict(checkpoint['model_state'])
        optimizer_g.load_state_dict(checkpoint['model_optim_state'])
        # current_step = checkpoint['counters']['current_step']
        # epoch = checkpoint['counters']['epoch']
        # checkpoint['sample_ts'].append(current_step)

        # torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=False)
        # latest = tf.train.latest_checkpoint(checkpoint_path)
        # model.saver.restore(sess, latest)
        print("Model loaded")
        #Visualize testset
        # Test_plot(model, offsetData['test'], poseData['test'], fileNames['test'], 'test', current_step, visibData['test'], miss_masks.offset_test,
        #         miss_masks.pose_test, scaler_pose, optimizer_g, injection_step)
    print('Process Ended!!!')

def create_model_Curric(shapedata, cfg, float_dtype, sampling=False):
    """Create translation model and initialize or load parameters."""

    model = seq2seq.Seq2SeqModel_Curriculum(
        shapedata,
        cfg
    )

    return model

def init_model(shapedata, float_dtype, cfg):
    model = create_model_Curric(shapedata, cfg, float_dtype)
    # if FLAGS.load == 0:
    model.apply(utils.init_weights)
    model.type(float_dtype).train()

    # model.train_writer.add_graph(sess.graph)
    if cfg.experiment.load == 0:
        with LoggingPrinter(settings.log_file): print("Model created")

    optimizer_g = optim.Adam(model.parameters(), lr=cfg.learning.learning_rate)

    if cfg.experiment.load == 1:
        #Read best checkpoint
        logger.info('Restoring from checkpoint {}'.format(cfg.model.ckpt_path))
        for f in os.listdir(cfg.model.ckpt_path):
            if f[0:15] == 'best_checkpoint':
                break
        checkpoint = torch.load(os.path.join(cfg.model.ckpt_path,f))
        # model = checkpoint['model_struct']
        model.load_state_dict(checkpoint['model_state'])
        optimizer_g.load_state_dict(checkpoint['model_optim_state'])
        current_step = checkpoint['counters']['current_step']
        epoch = checkpoint['counters']['epoch']
        m_ep = checkpoint['counters']['m_ep']
        m_st = checkpoint['counters']['m_st']
        checkpoint['sample_ts'].append(current_step)
    else:
        # Starting from scratch, so initialize checkpoint data structure
        current_step, epoch,m_ep, m_st = 0, 0, 0, 0
        checkpoint = {
            'losses': [],
            'losses_ts': [],
            'sample_ts': [],
            'norm_g': [],
            'counters': {
                'current_step': None,
                'epoch': None,
                'm_ep': None,
                'm_st': None
            },
            'model_state': None,
            'model_optim_state': None,
            'model_struct': None
        }

    return model, checkpoint, optimizer_g, current_step, epoch, m_ep, m_st
