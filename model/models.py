_author__ = "Carlo Seppi, Eva Schnider"
__copyright__ = "Copyright (C) 2020 Center for medical Image Analysis and Navigation"
__email__ = "carlo.seppi@unibas.ch"

import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as th
# from nnAudio import Spectrogram

import matplotlib.pyplot as plt


#####################################################################################################
####################   Nd:YAG #######################################################################
#####################################################################################################



#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
## important functions ##

def out_size(in_size, kernel_size, stride, padding=0, dilation=1):
    out_size = np.floor(((in_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)
    return out_size


def needed_padding(in_size, out_size, kernel_size, dilation=1):
    stride = 1
    padding = (out_size - in_size + dilation * (kernel_size - 1)) / 2
    return padding



#####################################################################################
## only initialize Conv weights ##

## should additonal weights be given? should it be seperate for FC and Conv?
@th.no_grad()
def init_weights_xavier_normal(m):  # this is also called the Glorot init.
    if type(m) == nn.Conv1d or  type(m) == nn.Conv2d or  type(m) == nn.Conv3d or type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)


@th.no_grad()
def init_weights_xavier_uniform(m):
    if type(m) == nn.Conv1d or  type(m) == nn.Conv2d or  type(m) == nn.Conv3d  or type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


@th.no_grad()
def init_weights_kaiming_uniform(m):  # this is also called the He init.
    if type(m) == nn.Conv1d or  type(m) == nn.Conv2d or  type(m) == nn.Conv3d  or type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight)


@th.no_grad()
def init_weights_kaiming_normal(m):
    if type(m) == nn.Conv1d or  type(m) == nn.Conv2d or  type(m) == nn.Conv3d  or type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)


###########################################################################################
###########################################################################################
###########################################################################################




###########################################################################################
###########################################################################################
###########################################################################################


class ParametricConvFCModel(nn.Module):

    def __init__(self, image, args, params):
        super(ParametricConvFCModel, self).__init__()


        ###################################################################################
        ###################################################################################
        ## initilize parameter for Cov-Layer from params ##
        channels_out_conv = [params['conv_layer_{}_channels_out'.format(i)] for i in range(1, params['n_conv_layers'] + 1)]

        ## order conv_out from small to big ##
        if params['conv_order']:
            channels_out_conv = sorted(channels_out_conv)


        strides = [params['conv_layer_{}_strides'.format(i)] for i in range(1, params['n_conv_layers'] + 1)]
        kernel_sizes = [params['conv_layer_{}_kernel_size'.format(i)] for i in range(1, params['n_conv_layers'] + 1)]
        kernel_sizes_maxpool = [params['conv_layer_{}_kernel_size_maxpool'.format(i)] for i in range(1, params['n_conv_layers'] + 1)]
        strides_maxpool = [params['conv_layer_{}_strides_maxpool'.format(i)] for i in range(1, params['n_conv_layers'] + 1)]

        ## Dropout and Batchnorm ##
        self.batchnorm_rate = []
        for i in range(1, params['n_conv_layers'] + 1):
                self.batchnorm_rate.append(params['conv_layer_{}_batchnorm'.format(i)]['name'])


        dropout_rates = [None for _ in channels_out_conv]
        for i in range(1, params['n_fc_layers'] + 1):
            if params['fc_layer_{}_extras'.format(i)]['name'] == 'dropout':
                dropout_rates.append(params['fc_layer_{}_extras'.format(i)]['rate'])
            else:
                dropout_rates.append(None)

            self.batchnorm_rate.append(params['fc_layer_{}_batchnorm'.format(i)]['name'])

        self.batchnorm_rate.append(False)

        self.layer_extras = [{'D': rate} if rate != None else None for rate in dropout_rates] +[None]
        ###################################################################################
        ###################################################################################
        ## Parameters ##

        input_channels, input_size = image.size()

        channels_out = channels_out_conv                  # all FC layer have same output: Output of Conv-Layer
        channels_in = [input_channels] + channels_out
        self.layer_types = ['C' for _ in range(params['n_conv_layers'])] + ['F' for _ in range(params['n_fc_layers']+1)]


        self.seq_layers = nn.ModuleDict()
        self.args = args
        self.params = params



        ######################################################################################################
        ######################################################################################################
        ## Activation Function ##

        if params['activation'] == 'relu':
            self.acti_func = F.relu
        elif params['activation'] == 'leakyrelu':
            self.acti_func = F.leaky_relu
        elif params['activation'] == 'elu':
            self.acti_func = F.elu
        elif params['activation'] == 'tanh':
            self.acti_func = th.tanh
        elif params['activation'] == 'sigmoid':
            self.acti_func = th.sigmoid
        else:
            raise ValueError('acivation function {} not supported'.format(params['activation']))

        ######################################################################################################
        ######################################################################################################
        ## Initual dustribution of the weights ##

        if params['init'] == 'kaiming_uniform':
            init_func = init_weights_kaiming_uniform
        elif params['init'] == 'kaiming_normal':
            init_func = init_weights_kaiming_normal
        elif params['init'] == 'xavier_normal':
            init_func = init_weights_xavier_normal
        elif params['init'] == 'xavier_uniform':
            init_func = init_weights_xavier_uniform
        elif params['init'] == 'standard':
            init_func = None
        else:
            raise ValueError('initialisation function {} not supported'.format(params['init']))



        ##################################################################################################
        ##################################################################################################
        ## Buildingstone of the Model ##
        j = 0
        # batch norm for input data
        if self.batchnorm_rate[0]:
            batchnorm = nn.BatchNorm1d(channels_in[0])
            self.seq_layers[str(j)] = batchnorm
            j = j + 1


        self.error = False
        for i in range(len(self.layer_types)):


            ##################################################################################################
            ##################################################################################################
            ## Convolutional Layer ##
            if self.layer_types[i] == 'C':
                size_out_conv = out_size(in_size=input_size, stride=strides[i], kernel_size=kernel_sizes[i])
                size_out_maxpool = out_size(in_size=size_out_conv, stride=strides_maxpool[i], kernel_size=kernel_sizes_maxpool[i])

                conv_layer = nn.Conv1d(in_channels=channels_in[i], out_channels=channels_out[i],
                                       kernel_size=kernel_sizes[i], stride=strides[i])


                input_size = size_out_maxpool
                self.seq_layers[str(j)] = conv_layer
                j = j + 1

                ## add batch norm after each conv ##
                if self.batchnorm_rate[i+1]:
                    batchnorm = nn.BatchNorm1d(channels_out[i])
                    self.seq_layers[str(j)] = batchnorm
                    j = j + 1


                ## maxpool ##
                maxpool = nn.MaxPool1d(kernel_sizes_maxpool[i],stride=strides_maxpool[i])
                self.seq_layers[str(j)] = maxpool
                j = j + 1


            ##################################################################################################
            ##################################################################################################
            ## FC Layer ##
            elif self.layer_types[i] == 'F':
                ## previous was convolution ##
                if self.layer_types[i - 1] == 'C':  # previous was convolution
                    out_channel = channels_out[i-1]
                    fc_input_size = int(out_channel * input_size)

                    #################################################
                    ## condition that model is invalid ##
                    if fc_input_size < args.fcinputsize[0] or fc_input_size > args.fcinputsize[1]:
                       print("\n*************************************")
                       print('Model is invalid, fc_input_size is {}, but should be between {} and {} according to your parameters file.'.format(fc_input_size, args.fcinputsize[1],args.fcinputsize[0]))
                       print('Change either the values for fcinputsize in the parameters file, or the size of your input.')
                       self.error = True
                       break


                if i != len(self.layer_types) - 1 and dropout_rates[i] != None:
                    ## add dropout
                    self.seq_layers[str(j)] = nn.Dropout(p=dropout_rates[i])
                    j = j + 1
                if i != len(self.layer_types) - 1:
                    ## FC same size as output of Conv
                    if fc_input_size>2000:  #reduces dimension
                        channels_out_ = 2000
                    else:
                        channels_out_ = fc_input_size
                else:
                    ## FC to classifyer
                    channels_out_ = 60 #len(self.args.classes)
                fully_con_layer = nn.Linear(fc_input_size, channels_out_)
                if fc_input_size>2000:  #reduces dimension
                        fc_input_size = 2000
                self.seq_layers[str(j)] = fully_con_layer
                j = j + 1

                ## add batchnorm on each layer, except last one
                if i != len(self.layer_types)-1 and self.batchnorm_rate[i+1]:
                    batchnorm = nn.BatchNorm1d(channels_out_)
                    self.seq_layers[str(j)] = batchnorm
                    j = j + 1




        ########################################################################################################
        ## inital weights for Conv Layer ##
        if init_func != None:
            self.seq_layers.apply(init_func)

    ##################################################################################################
    ##################################################################################################
    ## Define the Forward Model ##
    def forward(self, x):

        if self.error is True:
           x = -1
        else:


          self.inputs = x
          j=0

          if  self.batchnorm_rate[0]:
              layer = self.seq_layers[str(j)]
              x = layer(x)
              j += 1


          for i in range(len(self.layer_types)):

            ########################################################################################
            ########################################################################################
            ## Convolutional Layer ##
            if self.layer_types[i] == 'C':

                ## convolution layer ##
                layer = self.seq_layers[str(j)]
                x = layer(x)
                j += 1

                # for activation map #
                if i == 4:
                    self.grad_cam_feature_maps = x

                ## batchnorm ##
                if  self.batchnorm_rate[i+1]:
                    layer = self.seq_layers[str(j)]
                    x = layer(x)
                    j += 1

                ## activation layer ##
                x = self.acti_func(x)

                # # for activation map #
                # if i == 4:
                #     self.grad_cam_feature_maps = x

                ## maxpool ##
                layer = self.seq_layers[str(j)]
                x = layer(x)
                j = j+1


            ########################################################################################
            ########################################################################################
            ## FC Layer ##
            if self.layer_types[i] == 'F' and self.layer_types[i - 1] == 'C':
                ## switch from conv to fully connected ##
                x =  th.reshape(x, (x.size(0), -1))
                x_cnn = x
            if self.layer_extras[i] != None and 'D' in self.layer_extras[i]: # add dropout
                ## Dropout ##
                layer=self.seq_layers[str(j)]
                j += 1
                x=layer(x)

            if i < len(self.layer_types) - 1 and self.layer_types[i] != 'C':
                ## FC layer ##
                layer = self.seq_layers[str(j)]
                x = layer(x)
                j += 1

                ## batchnorm ##
                if  self.batchnorm_rate[i+1]:
                    layer = self.seq_layers[str(j)]
                    x = layer(x)
                    j += 1

                ## activation layer ##
                x = self.acti_func(x)

            elif i == len(self.layer_types) - 1:
                ## last layer ##
                layer = self.seq_layers[str(j)]
                x = layer(x)
#                x = F.softmax(x,dim=-1)
            ########################################################################################
            ########################################################################################






        # for activation map #
        self.classifications = th.argmax(x, dim=1)
        self.outputs = F.softmax(x,dim=1)

        return x, x_cnn


    ##################################################################################################
    ##################################################################################################
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
