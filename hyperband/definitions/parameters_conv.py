from hyperband.definitions.common_defs import *
from hyperopt import hp
from sklearn.metrics import roc_auc_score as AUC, log_loss, accuracy_score as accuracy
import torch as th
from torch.utils import data
import numpy as np
from data_loader import data_manager as dm
from model import models as m
from utils import utils
from parameter import parameter_hyperband as params_hyperband

## params for hyperparameter search are defined here: parameter/parameter_hyperband.py
space = params_hyperband.get_space()


def get_params():
    params = sample(space)
    return handle_integers(params)


#######################################################################################################
#######################################################################################################

def print_params(params):
    pprint({k: v for k, v in params.items() if not k.startswith('layer_')})
    print('')


def try_params(n_iterations, params, args_fixed, data_train, data_validate):
    n_epochs = n_iterations * 10  # one iteration equals 10 epochs
    print("epochs:", n_epochs)
    # print_params(params)

    ######################################################################################################
    ######################################################################################################
    ## load data with Data Manager ##

    # create data manager
    data_manager = dm.DataManager(args=args_fixed, params=params, data=data_train)  # csvpath=args_fixed.traincsv_path)
    #    data_manager_test = dm.DataManager(args=args_fixed, params=params, csvpath=args_fixed.testcsv_path)
    data_manager_val = dm.DataManager(args=args_fixed, params=params,
                                      data=data_validate)  # csvpath=args_fixed.validatecsv_path)

    # Parameters
    train_params = {'batch_size': params['batch_size'],
                    'shuffle': True,
                    'num_workers': args_fixed.nb_workers,
                    'pin_memory': True}
    test_params = {'batch_size': params['batch_size'],
                   'shuffle': False,
                   'num_workers': args_fixed.nb_workers,
                   'pin_memory': True}

    # data
    training_generator = data.DataLoader(data_manager, **train_params)
    #    training_generator_test = data.DataLoader(data_manager_test, **test_params)
    training_generator_val = data.DataLoader(data_manager_val, **test_params)

    ######################################################################################################
    ######################################################################################################

    #    input_tensor_size = next(iter(training_generator))[0].size()  # returns (batchsize, n_channels, length of signal)
    #    input_size = input_tensor_size[-1]
    #    input_channels = input_tensor_size[1] if len(input_tensor_size) > 2 else 1

    ######################################################################################################
    ######################################################################################################
    ## Model ##

    image, _ = data_manager_val[0]
    model = m.ParametricConvFCModel(image, args_fixed, params)

    ## check if model is valid ##
    if model.error is True:
        ## model is unvalid ##
        #        loss_value_mean_test = np.inf
        loss_value_mean_val = np.inf
    else:

        print_params(params)
        # seed = 10
        # th.manual_seed(seed)
        # np.random.seed(seed)

        ## model is valid ##
        #################################################################################################33
        #################################################################################################33

        ## loss function ##
        loss_func = utils.loss_function(params)

        ## optimizer ##
        optimizer = utils.choose_optimizer(args=args_fixed, params=params, model=model)

        print(model)
        # TRAINING
        device = th.device("cuda:" + str(args_fixed.gpu_id) if th.cuda.is_available() else "cpu")
        print(device)
        model.to(device)

        #################################################################################################33
        #################################################################################################33

        epoch_start = 0
        training_counter = 0

        for epoch in range(epoch_start, int(n_epochs)):
            model = model.train()

            #### Train ####
            model, optimizer, loss_value_mean = utils.train_network(model, optimizer, loss_func, device,
                                                                    training_generator, params)
            print("Train mean loss: {}".format(loss_value_mean))

        ## VALIDATION ##
        #        _ , _ , _ , loss_value_mean_test = utils.test_network(model,loss_func,device,args_fixed,training_generator_test)
        #        print("Test mean loss: {}".format(loss_value_mean_test))
        loss_value_mean_val, _, _ = utils.test_network(model, loss_func, device, args_fixed, training_generator_val)
        print("Validation mean loss: {}".format(loss_value_mean_val))

    #    loss = (loss_value_mean_test+loss_value_mean_val)/2
    loss = loss_value_mean_val
    #    print('Test mean loss: {}, Validation mean loss: {}, MEAN loss: {}'.format(loss_value_mean_test,loss_value_mean_val,loss))
    print('Validation mean loss: {}, MEAN loss: {}'.format(loss_value_mean_val, loss))

    return {'loss': loss}
