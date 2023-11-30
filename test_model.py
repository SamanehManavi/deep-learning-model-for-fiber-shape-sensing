import torch as th

from data_loader import data_manager as dm
from torch.utils import data
from model import models as m
from utils import utils
from parameter import run_args
import pathlib
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# loading_from_folder='30minTrain_3minTest_3minVal_100p_rep2'
# print('Model is loading from ...')
# print(loading_from_folder)

def infer_and_evaluate(args, state, chose_path):
    params = state['params']
    ## loss function
    loss = state['loss']

    ######################################################################################################
    ######################################################################################################
    ## load data with Data Manager ##

    print("Load Data...")
    data_test = dm.LoadData(args=args, csvpath=chose_path)
    #print(data_test.shape)
    print("Done.")

    data_manager_test = dm.DataManager(args=args, params=params, data=data_test)
    params_test = {'batch_size': params['batch_size'],
                   'shuffle': False,
                   'num_workers': args.nb_workers,
                   'pin_memory': True}

    training_generator_test = data.DataLoader(data_manager_test, **params_test)

    ######################################################################################################
    ######################################################################################################
    ## Model ##

    image, _ = data_manager_test[0]
    model = m.ParametricConvFCModel(image, args, params)

    print(model)
    model.load_state_dict(state['network'])

    print("******************** CUDA ***********************")
    device = th.device("cuda:" + str(args.gpu_id) if th.cuda.is_available() else "cpu")
    model.to(device)
    print(device)

    ##########################################################################################################
    ##########################################################################################################
    #### Evaluate ####
    # loss_value_mean,
    loss_value_mean, output, label = utils.test_network(model, loss, device, args, training_generator_test)
    # print("mean loss: {}".format(loss_value_mean))

    pathlib.Path(args.path_to_folder+'/results/gradcam').mkdir(parents=True, exist_ok=True)
    np.savetxt(args.path_to_folder+"/results/gradcam/output.txt", output.numpy().astype(float), fmt='%f')
    np.savetxt(args.path_to_folder+"/results/gradcam/label.txt", label.numpy().astype(float), fmt='%f')


if __name__ == "__main__":

    args = run_args.parser.parse_args()
    dataset_for_inference = args.infer_data
    checkpoint_to_load = args.infer_model

    #dataset_for_inference = 'test'
    # checkpoint_to_load = 'best'

    if dataset_for_inference == 'train':
        chose_path = args.traincsv_path
    elif dataset_for_inference == 'test':
        chose_path = args.testcsv_path
    elif dataset_for_inference == 'validate':
        chose_path = args.validatecsv_path
    else:
        raise NotImplementedError

    if checkpoint_to_load == 'last':
        path = args.path_to_folder+'/results/model/model_last.pt'
        # path = './results/model/'+loading_from_folder+'/model_last.pt'
        state = th.load(path)
        args = state['args']
        mod_model = 'last'
    elif checkpoint_to_load == 'best':
        # path = './results/model/'+loading_from_folder+'/model_best.pt'
        path = args.path_to_folder+'/results/model/model_best.pt'
        state = th.load(path)
        args = state['args']
        mod_model = 'best'
    else:
        raise NotImplementedError

    print("Infer and evaluate on {} data, using the {} checkpoint".format(dataset_for_inference, checkpoint_to_load))
    print("change data using the --infer_data flag to train, validate or test")
    print("change checkpoint using the --infer_model flag to last or best")

    epoch_start = state['epoch']
    print("****************************************************************")
    print("************* epoch ", epoch_start, ", ***********************")
    print("************* path ", chose_path, ", ***********************")
    print("****************************************************************")

    # print(chose_path)
    print(args.path)
    infer_and_evaluate(args, state, chose_path)
