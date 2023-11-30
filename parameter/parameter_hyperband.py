__author__ = "Carlo Seppi"
__copyright__ = "Copyright (C) 2020 Center for medical Image Analysis and Navigation"
__email__ = "carlo.seppi@unibas.ch"

from hyperopt import hp



def get_space():

        # parameters -- [min,max,step]
        # channel_out_range = [1,512,1]
        # kernel_size_range = [1,25,1]
        layers_conv_range = [1,20,1]
        layers_fc_range = [1,20,1]
        stride_range = [1,2,1]
        maxpool_range = [2,3,1]
        maxpool_stride_range = [1,2,1]
        space = {
            'n_conv_layers': hp.quniform('cov', layers_conv_range[0], layers_conv_range[1], layers_conv_range[2]),
            'n_fc_layers': hp.quniform('fc', layers_fc_range[0], layers_fc_range[1], layers_fc_range[2]),
            'activation': 'relu', #hp.choice('a', ('sigmoid','relu', 'tanh', 'elu', 'leakyrelu')),
            'init': hp.choice('i', ('standard', 'xavier_uniform','xavier_normal', 'kaiming_uniform', 'kaiming_normal')),
            'batch_size': 256,  #hp.choice('bs', (32,64, 128)),
            'optimizer': 'Adam', #hp.choice('o', ('RMSprop', 'Rprop', 'SGD', 'Adam')),
            'learnin_rate':   hp.choice('bs', (0.01, 0.001,0.0001,0.00001)),     # Leanring rate lr
            'conv_order': hp.choice('co',(True,False)),                 # True sort from small to big, False no sorting
            'L2_reguralization': hp.choice('bs', (0.1, 0.01, 0.001,0.0001,0.00001,0)),
            'beta': hp.uniform('beta', 0.0, 5.0),
        }
            # 'margin':  hp.choice('m', (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)), #hp.uniform('m', 0.,1.),
            # 'reg_alpha':  hp.choice('a', (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)),

        # for each hidden layer, we choose size, activation and extras individually
        for i in range(1, layers_conv_range[1] + 1):
            space['conv_layer_{}_channels_out'.format(i)] = hp.choice('ls{}'.format(i), (2,4,8,16,32,64,128,256,512))  #hp.quniform('ls{}'.format(i), channel_out_range[0],channel_out_range[1],channel_out_range[2])
            space['conv_layer_{}_kernel_size'.format(i)] = 3 #hp.quniform('ls{}'.format(i), kernel_size_range[0],kernel_size_range[1],kernel_size_range[2])
            space['conv_layer_{}_extras'.format(i)] = hp.choice('e{}'.format(i), ({'name': None}))
            space['conv_layer_{}_strides'.format(i)] = hp.quniform('ls{}'.format(i), stride_range[0],stride_range[1],stride_range[2])
            space['conv_layer_{}_kernel_size_maxpool'.format(i)] = hp.quniform('mp{}'.format(i), maxpool_range[0],maxpool_range[1],maxpool_range[2])
            space['conv_layer_{}_strides_maxpool'.format(i)] = hp.quniform('mp{}'.format(i), maxpool_stride_range[0],maxpool_stride_range[1],maxpool_stride_range[2])
            space['conv_layer_{}_batchnorm'.format(i)] = hp.choice('e{}'.format(i), ({'name': True},{'name': False}))


        for i in range(1, layers_fc_range[1]+1):
            space['fc_layer_{}_extras'.format(i)] = hp.choice('e{}'.format(i), (
                {'name': 'dropout', 'rate': hp.uniform('d{}'.format(i), 0.1, 0.8)},{'name': None}))
            space['fc_layer_{}_batchnorm'.format(i)] = hp.choice('e{}'.format(i), ({'name': True},{'name': False}))

        return space
