# import EarlyStopping
from utils.pytorchtools import EarlyStopping

import pathlib
import sys
import time

import numpy as np
import torch as th
from torch.utils import data

import hyperband.definitions.common_defs as cc
from data_loader import data_manager as dm
from model import models as m
from parameter import run_args
from utils import utils

import setproctitle as SP
SP.setproctitle('FiberSensor')
SP.setthreadtitle('FiberSensor')

def print_params(params):
    cc.pprint({k: v for k, v in params.items() if not k.startswith('layer_')})
    print('')


def train(args, epoch_start, state):
    patience=5000
    if epoch_start > 0:
        params = state['params']
    else:
        params ={'L2_reguralization': 0,
 'activation': 'relu',
 'batch_size': 256,
 'beta': 4.043486020840367,
 'conv_layer_10_batchnorm': {'name': True},
 'conv_layer_10_channels_out': 4,
 'conv_layer_10_extras': 'name',
 'conv_layer_10_kernel_size': 3,
 'conv_layer_10_kernel_size_maxpool': 3,
 'conv_layer_10_strides': 1,
 'conv_layer_10_strides_maxpool': 1,
 'conv_layer_11_batchnorm': {'name': True},
 'conv_layer_11_channels_out': 2,
 'conv_layer_11_extras': 'name',
 'conv_layer_11_kernel_size': 3,
 'conv_layer_11_kernel_size_maxpool': 2,
 'conv_layer_11_strides': 1,
 'conv_layer_11_strides_maxpool': 2,
 'conv_layer_12_batchnorm': {'name': False},
 'conv_layer_12_channels_out': 256,
 'conv_layer_12_extras': 'name',
 'conv_layer_12_kernel_size': 3,
 'conv_layer_12_kernel_size_maxpool': 2,
 'conv_layer_12_strides': 1,
 'conv_layer_12_strides_maxpool': 2,
 'conv_layer_13_batchnorm': {'name': True},
 'conv_layer_13_channels_out': 64,
 'conv_layer_13_extras': 'name',
 'conv_layer_13_kernel_size': 3,
 'conv_layer_13_kernel_size_maxpool': 3,
 'conv_layer_13_strides': 2,
 'conv_layer_13_strides_maxpool': 2,
 'conv_layer_14_batchnorm': {'name': False},
 'conv_layer_14_channels_out': 64,
 'conv_layer_14_extras': 'name',
 'conv_layer_14_kernel_size': 3,
 'conv_layer_14_kernel_size_maxpool': 2,
 'conv_layer_14_strides': 1,
 'conv_layer_14_strides_maxpool': 2,
 'conv_layer_15_batchnorm': {'name': True},
 'conv_layer_15_channels_out': 512,
 'conv_layer_15_extras': 'name',
 'conv_layer_15_kernel_size': 3,
 'conv_layer_15_kernel_size_maxpool': 3,
 'conv_layer_15_strides': 2,
 'conv_layer_15_strides_maxpool': 1,
 'conv_layer_16_batchnorm': {'name': False},
 'conv_layer_16_channels_out': 8,
 'conv_layer_16_extras': 'name',
 'conv_layer_16_kernel_size': 3,
 'conv_layer_16_kernel_size_maxpool': 2,
 'conv_layer_16_strides': 2,
 'conv_layer_16_strides_maxpool': 2,
 'conv_layer_17_batchnorm': {'name': False},
 'conv_layer_17_channels_out': 128,
 'conv_layer_17_extras': 'name',
 'conv_layer_17_kernel_size': 3,
 'conv_layer_17_kernel_size_maxpool': 2,
 'conv_layer_17_strides': 1,
 'conv_layer_17_strides_maxpool': 2,
 'conv_layer_18_batchnorm': {'name': False},
 'conv_layer_18_channels_out': 256,
 'conv_layer_18_extras': 'name',
 'conv_layer_18_kernel_size': 3,
 'conv_layer_18_kernel_size_maxpool': 3,
 'conv_layer_18_strides': 2,
 'conv_layer_18_strides_maxpool': 2,
 'conv_layer_19_batchnorm': {'name': True},
 'conv_layer_19_channels_out': 256,
 'conv_layer_19_extras': 'name',
 'conv_layer_19_kernel_size': 3,
 'conv_layer_19_kernel_size_maxpool': 2,
 'conv_layer_19_strides': 1,
 'conv_layer_19_strides_maxpool': 2,
 'conv_layer_1_batchnorm': {'name': True},
 'conv_layer_1_channels_out': 32,
 'conv_layer_1_extras': 'name',
 'conv_layer_1_kernel_size': 3,
 'conv_layer_1_kernel_size_maxpool': 3,
 'conv_layer_1_strides': 1,
 'conv_layer_1_strides_maxpool': 2,
 'conv_layer_20_batchnorm': {'name': False},
 'conv_layer_20_channels_out': 32,
 'conv_layer_20_extras': 'name',
 'conv_layer_20_kernel_size': 3,
 'conv_layer_20_kernel_size_maxpool': 3,
 'conv_layer_20_strides': 2,
 'conv_layer_20_strides_maxpool': 2,
 'conv_layer_2_batchnorm': {'name': False},
 'conv_layer_2_channels_out': 16,
 'conv_layer_2_extras': 'name',
 'conv_layer_2_kernel_size': 3,
 'conv_layer_2_kernel_size_maxpool': 2,
 'conv_layer_2_strides': 1,
 'conv_layer_2_strides_maxpool': 1,
 'conv_layer_3_batchnorm': {'name': False},
 'conv_layer_3_channels_out': 32,
 'conv_layer_3_extras': 'name',
 'conv_layer_3_kernel_size': 3,
 'conv_layer_3_kernel_size_maxpool': 3,
 'conv_layer_3_strides': 1,
 'conv_layer_3_strides_maxpool': 2,
 'conv_layer_4_batchnorm': {'name': False},
 'conv_layer_4_channels_out': 16,
 'conv_layer_4_extras': 'name',
 'conv_layer_4_kernel_size': 3,
 'conv_layer_4_kernel_size_maxpool': 3,
 'conv_layer_4_strides': 2,
 'conv_layer_4_strides_maxpool': 1,
 'conv_layer_5_batchnorm': {'name': False},
 'conv_layer_5_channels_out': 256,
 'conv_layer_5_extras': 'name',
 'conv_layer_5_kernel_size': 3,
 'conv_layer_5_kernel_size_maxpool': 2,
 'conv_layer_5_strides': 1,
 'conv_layer_5_strides_maxpool': 2,
 'conv_layer_6_batchnorm': {'name': False},
 'conv_layer_6_channels_out': 128,
 'conv_layer_6_extras': 'name',
 'conv_layer_6_kernel_size': 3,
 'conv_layer_6_kernel_size_maxpool': 3,
 'conv_layer_6_strides': 1,
 'conv_layer_6_strides_maxpool': 1,
 'conv_layer_7_batchnorm': {'name': False},
 'conv_layer_7_channels_out': 16,
 'conv_layer_7_extras': 'name',
 'conv_layer_7_kernel_size': 3,
 'conv_layer_7_kernel_size_maxpool': 2,
 'conv_layer_7_strides': 1,
 'conv_layer_7_strides_maxpool': 1,
 'conv_layer_8_batchnorm': {'name': False},
 'conv_layer_8_channels_out': 64,
 'conv_layer_8_extras': 'name',
 'conv_layer_8_kernel_size': 3,
 'conv_layer_8_kernel_size_maxpool': 2,
 'conv_layer_8_strides': 1,
 'conv_layer_8_strides_maxpool': 2,
 'conv_layer_9_batchnorm': {'name': True},
 'conv_layer_9_channels_out': 256,
 'conv_layer_9_extras': 'name',
 'conv_layer_9_kernel_size': 3,
 'conv_layer_9_kernel_size_maxpool': 3,
 'conv_layer_9_strides': 1,
 'conv_layer_9_strides_maxpool': 1,
 'conv_order': True,
 'fc_layer_10_batchnorm': {'name': False},
 'fc_layer_10_extras': {'name': 'dropout', 'rate': 0.6786420318713986},
 'fc_layer_11_batchnorm': {'name': True},
 'fc_layer_11_extras': {'name': 'dropout', 'rate': 0.7280264965067813},
 'fc_layer_12_batchnorm': {'name': True},
 'fc_layer_12_extras': {'name': 'dropout', 'rate': 0.3109104545086341},
 'fc_layer_13_batchnorm': {'name': True},
 'fc_layer_13_extras': {'name': 'dropout', 'rate': 0.6340271613087337},
 'fc_layer_14_batchnorm': {'name': False},
 'fc_layer_14_extras': {'name': 'dropout', 'rate': 0.3472826133964033},
 'fc_layer_15_batchnorm': {'name': False},
 'fc_layer_15_extras': {'name': None},
 'fc_layer_16_batchnorm': {'name': True},
 'fc_layer_16_extras': {'name': 'dropout', 'rate': 0.7913542346706879},
 'fc_layer_17_batchnorm': {'name': False},
 'fc_layer_17_extras': {'name': 'dropout', 'rate': 0.43902082991627256},
 'fc_layer_18_batchnorm': {'name': True},
 'fc_layer_18_extras': {'name': None},
 'fc_layer_19_batchnorm': {'name': True},
 'fc_layer_19_extras': {'name': 'dropout', 'rate': 0.5106751029005038},
 'fc_layer_1_batchnorm': {'name': True},
 'fc_layer_1_extras': {'name': None},
 'fc_layer_20_batchnorm': {'name': False},
 'fc_layer_20_extras': {'name': None},
 'fc_layer_2_batchnorm': {'name': True},
 'fc_layer_2_extras': {'name': 'dropout', 'rate': 0.36604521112446964},
 'fc_layer_3_batchnorm': {'name': False},
 'fc_layer_3_extras': {'name': None},
 'fc_layer_4_batchnorm': {'name': True},
 'fc_layer_4_extras': {'name': None},
 'fc_layer_5_batchnorm': {'name': False},
 'fc_layer_5_extras': {'name': 'dropout', 'rate': 0.16332097814130914},
 'fc_layer_6_batchnorm': {'name': False},
 'fc_layer_6_extras': {'name': None},
 'fc_layer_7_batchnorm': {'name': False},
 'fc_layer_7_extras': {'name': 'dropout', 'rate': 0.2259942771046244},
 'fc_layer_8_batchnorm': {'name': True},
 'fc_layer_8_extras': {'name': None},
 'fc_layer_9_batchnorm': {'name': True},
 'fc_layer_9_extras': {'name': 'dropout', 'rate': 0.3971770529901033},
 'init': 'xavier_normal',
 'learnin_rate': 0.0001,
 'n_conv_layers': 5,
 'n_fc_layers': 5,
 'optimizer': 'Adam'}

    # seed = 10
    # random.seed(seed)
    # th.manual_seed(seed)
    # np.random.seed(seed)

    print_params(params)

    ######################################################################################################
    ######################################################################################################
    ## load data with Data Manager ##


    print("Load Data...")
    data_train = dm.LoadData(args=args, csvpath=args.traincsv_path)
    data_validate = dm.LoadData(args=args, csvpath=args.validatecsv_path)
    print("************* training_path ", args.traincsv_path, ", ***********************")
    print("************* validation_path ", args.validatecsv_path, ", ***********************")

    print("Done.")

    # create data manager
    data_manager = dm.DataManager(args=args, params=params, data=data_train)
    data_manager_validate = dm.DataManager(args=args, params=params, data=data_validate)

    # Parameters
    train_params = {'batch_size': params['batch_size'],
                    'shuffle': True,
                    'num_workers': args.nb_workers,
                    'pin_memory': False}
    validate_params = {'batch_size': params['batch_size'],
                       'shuffle': False,
                       'num_workers': args.nb_workers,
                       'pin_memory': False}

    # data
    training_generator = data.DataLoader(data_manager, **train_params)
    training_generator_validate = data.DataLoader(data_manager_validate, **validate_params)

    ######################################################################################################
    ######################################################################################################
    ## Model ##

    device = th.device("cuda:" + str(args.gpu_id) if th.cuda.is_available() else "cpu")
    image, _ = data_manager_validate[0]
    model = m.ParametricConvFCModel(image, args, params)

    #################################################################################################33
    #################################################################################################33
    ## Check if model is valid"

    print(device)
    print(model)

    if model.error is True:
        print("Error: Change parameters!")
        print("*************************************")
    else:

        #################################################################################################33
        #################################################################################################33
        ## load network and initilize parameters ##
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=patience, verbose=True)

        if epoch_start > 0:
            ## Use old Network: load parameters and weights of network ##
            model.load_state_dict(state['network'])
            best_mean_loss = state['best_mean_loss']
            loss = state['loss']
        else:
            ## Use New Network: initialize parameters of network ##
            best_mean_loss = 1e10
            # loss function
            loss = utils.loss_function(params)

        ## optimizer ##
        optimizer = utils.choose_optimizer(args=args, params=params, model=model)
        #################################################################################################33
        #################################################################################################33

        model.to(device)
        train_counter = 0
        Train_History=[]
        Val_History=[]
        for epoch in range(epoch_start, args.maxepoch):

            print("****************************************************")
            print("************* epoch ", epoch, "***************************")
            print("****************************************************")
            #### Train ####
            model, optimizer, loss_value_mean = utils.train_network(model, optimizer, loss, device, training_generator, params)
            Train_History.append(loss_value_mean)
            print("mean loss: {}".format(loss_value_mean))

            #### Validate after each epoch ####
            print("***************************")
            print("********* Validate ************")
            print("***************************")
            loss_value_mean_validate, _, _ = utils.test_network(model, loss, device, args, training_generator_validate)
            Val_History.append(loss_value_mean_validate)


            print("mean loss: {}".format(loss_value_mean_validate))

            ##################################################################################
            ##################################################################################

            # ## Do a weighted accuracy -- makes sense, if distribution of classes is unequal ##
            # weighted_mean_accuracy = 0
            # for i in range(len(args.classes)):
            #     weighted_mean_accuracy += 100 * class_correct["array"][i] / class_total["array"][i]
            # weighted_mean_accuracy = weighted_mean_accuracy / len(args.classes)

            ## save and print ##
            # print('************* Validate DATA *******************')
            # print('Accuracy of the network on the Validation images: %.5f %%' % (weighted_mean_accuracy))
            # np.savetxt(args.path_to_folder + "/results/model/class_predict_last", class_correct["matrix"].astype(int), fmt='%d')

            ## print confusion matrix ##
            # for i in range(len(args.classes)):
            #     print('Accuracy of %5s : %.5f %%' % (
            #         args.classes[i], 100 * class_accuracy["array"][i]))

            ##################################################################################
            ##################################################################################

            ## save last and best state ##

            ## we use mean_accuracy as measurment ##
            mean_loss_overall = loss_value_mean_validate

            state = {
                'train_counter': train_counter,
                'args': args,
                'network': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_mean_loss': best_mean_loss,
                'params': params,
                'loss': loss}

            if best_mean_loss > mean_loss_overall:
                best_mean_loss = mean_loss_overall
                state['best_mean_loss'] = best_mean_loss
                utils.save_checkpoint(state, args.path_to_folder+'/results/model', "model_best.pt")
                # np.savetxt(args.path_to_folder+'/results/model' + "/class_predict_best",
                           # class_correct["matrix"].astype(int), fmt='%d')

            utils.save_checkpoint(state, args.path_to_folder+'/results/model', "model_last.pt")
            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(mean_loss_overall)

            if early_stopping.early_stop:
                print("Early stopping")
                break
            ##################################################################################
            ##################################################################################
    return Train_History, Val_History

if __name__ == "__main__":

    input_args = run_args.parser.parse_args()

    if input_args.mode == 'continue':
        print("Continue Training")
        path = args.path_to_folder+'/results/model/model_best.pt'#args.path_to_folder+
        state = th.load(path)
        args = state['args']
        epoch_start = state['epoch'] + 1
    elif input_args.mode == 'restart':
        print("Restart Training")
        args = input_args
        epoch_start = 0
        state = 0
    else:
        raise NotImplementedError('unknown mode')

    ## create folder, if they don't exist ##
    pathlib.Path(args.path_to_folder+'/results/output').mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.path_to_folder+'/results/model').mkdir(parents=True, exist_ok=True)

    ## write all the output in log file ##
    # if input_args.logfile is True:
    time_string = time.strftime("%Y%m%d-%H%M%S")
    log_file_path = args.path_to_folder+'/results/output/output_{}.log'.format(time_string)
    print("Check log file in: ")
    print(log_file_path)
    sys.stdout = open(log_file_path, 'w')
    # else:
    #     print("No log file, only print to console")

    Train_History, Val_History = train(args, epoch_start, state)
    np.savetxt(args.path_to_folder+"/results/model/Train_History.txt", Train_History, delimiter=";")
    np.savetxt(args.path_to_folder+"/results/model/Val_History.txt", Val_History, delimiter=";")
