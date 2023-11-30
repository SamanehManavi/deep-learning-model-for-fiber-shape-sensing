import torch.nn.functional as F



import os
import torch as th
import scipy.io
import numpy as np
import scipy.fftpack as ft
# import matplotlib.pyplot as plt
# import tkinter



#####################################################################################
#####################################################################################
def save_checkpoint(state, path, filename):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=False)
    th.save(state, os.path.join(path, filename))



def loss_function(params):
    loss=th.nn.SmoothL1Loss(beta=params["beta"]) #size_average=None, reduce=None, reduction='mean', beta=1.0)
    #loss = th.nn.CrossEntropyLoss()
    return loss


#####################################################################################
#####################################################################################


## programm the traing part of the network ##
def train_network(model,optimizer,loss,device,training_generator,params):

      ## Output:
      #  model
      #  mean_loss_value

      model = model.train()
      loss_value_mean = 0.
      total_number = 0
      train_counter = 0
      for spec, target_label in training_generator:

            if train_counter == 0:
                batch_size = target_label.size(0)

            optimizer.zero_grad()
            model.zero_grad()
            input = spec.to(device)
            target_label = target_label.to(device)

            # net_output, cnn_output = model(input)
            ## create siamese differents inside the batches ##
#            rand  = th.randperm(cnn_output.shape[0])
#            cnn_output_siamese  = cnn_output[rand,:].clone().detach()

#            target_label_siamese  = target_label[rand,:].clone().detach()

            ## D = ||G(X1)-G(X2)||_2
#            cnn_diff  =  th.sqrt(th.sum((cnn_output-cnn_output_siamese)**2,dim=-1))
#            cnn_diff2  = th.sqrt(th.sum((cnn_output-cnn_output_siamese2)**2,dim=-1))
            ## cos simularity   u*v/||u||||v||
#            cnn_diff  = F.cosine_similarity(cnn_output,cnn_output_siamese)

#            diff  = th.sqrt(th.sum((target_label-target_label_siamese)**2,dim=-1))
#            loss_cnn = 0.
#            margin = params["margin"]
#            for i in range(cnn_diff.shape[0]):
#                Dn = cnn_diff[i]
#                margin = 1./1000*diff[i]
#                loss_cnn += max(0.,-Dn+margin)/cnn_diff.shape[0]
#            loss_cnn.backward()
            # loss_value = params["reg_alpha"]*loss(net_output, target_label) #+(1-params["reg_alpha"])*loss_cnn

            net_output, _ = model(input)
            loss_value = loss(net_output, target_label)

            loss_value.backward()
            optimizer.step()

            ## print loss value ##
            ## print only ~6 loss values
            if train_counter % np.floor(len(training_generator)/5) == 0:
               print("loss function at mini-batch iteration {} : {}".format(train_counter, loss_value.item()))
            train_counter += 1

            ## get the mean loss over all
            loss_value_mean += loss_value.item()*target_label.size(0)
            total_number += target_label.size(0)

      loss_value_mean /= total_number
      return model, optimizer, loss_value_mean



## programm the traing part of the network ##
def test_network(model,loss,device,args,training_generator_test):

    ## Output:
    # class_correct     --  correct prediction
    # class_total       --  total true distribution
    # class_accuracy    --  accuracy
    #   - matrix        --  confucius matrix
    #   - array         --  represent it as array
    #   - single        --  all in an unweighted single value
    # loss_value_mean   --  mean loss



    with th.no_grad():
        model = model.eval()

        loss_value_mean = 0
        total_number = 0

        correct = 0
        total = 0
        output = th.tensor([])
        label  = th.tensor([])
        for spec, target_label in training_generator_test:

            input = spec.to(device)
            target_label = target_label.to(device)

            output_cnn, _ = model(input)
            loss_value = loss(output_cnn, target_label)

            loss_value_mean += loss_value.item()*target_label.size(0)
            total_number += target_label.size(0)


            output = th.cat((output, output_cnn.to("cpu")), 0)
            label  = th.cat((label, target_label.to("cpu")), 0)

    loss_value_mean /= total_number
    return loss_value_mean, output, label

# def test_network_2(model,loss,device,args,training_generator_test):
#
#     ## Output:
#     # class_correct     --  correct prediction
#     # class_total       --  total true distribution
#     # class_accuracy    --  accuracy
#     #   - matrix        --  confucius matrix
#     #   - array         --  represent it as array
#     #   - single        --  all in an unweighted single value
#     # loss_value_mean   --  mean loss
#
#
#
#     with th.no_grad():
#         model = model.eval()
#
#         loss_value_mean = 0
#         total_number = 0
#
#         correct = 0
#         total = 0
#         output = th.tensor([])
#         label  = th.tensor([])
#         for spec, target_label in training_generator_test:
#
#             input = spec.to(device)
#             target_label = target_label.to(device)
#
#             output_cnn, _ = model(input)
#             loss_value = loss(output_cnn, target_label)
#
#             loss_value_mean += loss_value.item()*target_label.size(0)
#             total_number += target_label.size(0)
#
#
#             output = th.cat((output, output_cnn.to("cpu")), 0)
#             label  = th.cat((label, target_label.to("cpu")), 0)
#
#     loss_value_mean /= total_number
#     return loss_value_mean, output, label





#####################################################################################
#####################################################################################
## choose optimizer ##
def choose_optimizer(args,params,model):
      optimizer = None
      if params['optimizer'] == 'RMSprop':
          optimizer = th.optim.RMSprop(model.parameters(), lr=params['learnin_rate'],weight_decay=params['L2_reguralization'])
      elif params['optimizer'] == 'Adam':
          optimizer = th.optim.Adam(model.parameters(), lr=params['learnin_rate'], amsgrad=args.amsgrad,weight_decay=params['L2_reguralization']) #additional parameter? amsgrad?
      elif params['optimizer'] == 'Rprop':
          optimizer = th.optim.Rprop(model.parameters(), lr=params['learnin_rate'])
      elif params['optimizer'] == 'SGD':
          optimizer = th.optim.SGD(model.parameters(), lr=params['learnin_rate'],weight_decay=params['L2_reguralization'])

      return optimizer
