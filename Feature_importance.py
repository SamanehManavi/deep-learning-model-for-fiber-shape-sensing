import matplotlib.pyplot as plt
import numpy as np
import torch as th
# import torch.nn.functional as F
import pathlib
# from scipy.ndimage import gaussian_filter
# from skimage import transform
from torch.utils import data
from data_loader import data_manager as dm
from model import models as m
from parameter import run_args as param
from utils import utils
import itertools
from torch import linalg as LA
# from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
# import os
import matplotlib
from matplotlib import font_manager
matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
font_dirs = ["./Times_New_Roman"]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
custom_font_manager = font_manager.FontManager()
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
plt.rcParams["font.family"] = "Times New Roman"
font = {'size'   : 24}
matplotlib.rc('font', **font)

Col=['#1b9e77','#d95f02','#7570b3','#9e1b84','#a6611a','#d01c8b','#2c7bb6']
# fig=plt.figure(figsize=(20, 10))#26, 13
# plt.ion()


loading_from_folder = 'RandomTest_1606'
# print('Model is loading from ...')
# print(loading_from_folder)


def relative_absolute(pred_Relative):
    pred_Absolute = th.zeros((len(pred_Relative), 60))
    for i in range(0, len(pred_Relative)):
        pred_Absolute[i, 0:3] = pred_Relative[i, 0:3]
        for j in range(1, 20):
            pred_Absolute[i, 3 * j] = pred_Absolute[i, 3 * j - 3] + pred_Relative[i, 3 * j]
            pred_Absolute[i, 3 * j + 1] = pred_Absolute[i, 3 * j - 2] + pred_Relative[i, 3 * j + 1]
            pred_Absolute[i, 3 * j + 2] = pred_Absolute[i, 3 * j - 1] + pred_Relative[i, 3 * j + 2]
    return pred_Absolute


def absolute_relative(pred_Absolute):
    pred_Relative = th.zeros((1, 60))
    pred_Relative[0, 0:3] = pred_Absolute[0, 0:3]
    for j in range(0, 19):
        pred_Relative[0, 3 * j + 3] = pred_Absolute[0, 3 * j + 3] - pred_Absolute[0, 3 * j]
        pred_Relative[0, 3 * j + 4] = pred_Absolute[0, 3 * j + 4] - pred_Absolute[0, 3 * j + 1]
        pred_Relative[0, 3 * j + 5] = pred_Absolute[0, 3 * j + 5] - pred_Absolute[0, 3 * j + 2]
    return pred_Relative



def grad_cam(args_fixed, state, chose_path):

    params = state['params']

    ## loss function
    loss = state['loss']

    ######################################################################################################
    ######################################################################################################
    ## load data with Data Manager ##

    # data_manager_test = dm.DataManager(args=args_fixed, params=params, csvpath=chose_path)
    print("Load Data...")
    data_test = dm.LoadData(args=args, csvpath=chose_path)
    print(data_test.data.shape)
    print("Done.")

    spectra = th.mean(data_test.data, 2)
    STD_spectra = th.std(spectra, 0)
    print(STD_spectra.shape)


    data_manager_test = dm.DataManager(args=args, params=params, data=data_test)

    params_test = {'batch_size': 1,  # params['batch_size'],
                   'shuffle': False,
                   'num_workers': args.nb_workers,
                   'pin_memory': True}

    training_generator_test = data.DataLoader(data_manager_test, **params_test)

    ######################################################################################################
    ######################################################################################################
    ## Model ##

    # device = th.device("cuda:" + str(args_fixed.gpu_id) if th.cuda.is_available() else "cpu")
    device = th.device("cuda:" + str(args.gpu_id) if th.cuda.is_available() else "cpu")
    image, _ = data_manager_test[0]
    print(image.shape)
    model = m.ParametricConvFCModel(image, args_fixed, params)

    print(model)
    model.load_state_dict(state['network'])

    for mm in model.modules():
        if isinstance(mm, th.nn.BatchNorm1d) or isinstance(mm, th.nn.BatchNorm2d):
            mm.eval()

    ## optimizer ##
    optimizer = utils.choose_optimizer(args=args_fixed, params=params, model=model)

    print("******************** CUDA ***********************")
    model.to(device)
    print(device)

    ##########################################################################################################
    ##########################################################################################################
    #### Evaluate ####

    count = 0

    # print(len(training_generator_test.dataset))
    # print(my_sample)
    # y_sn_k = np.ones((1, 190, 60))
    wavelength = np.linspace(800, 890, num=190)
    peaks = np.array([812.5, 816.4, 820.4, 824.5, 828.5, 832.3, 836.3, 840.2, 844.2, 848.0, 852.0, 856.2, 860.0, 863.9, 868.1])
    # RM = np.array([[0.00629614, 0.01032155, -0.99992608], [0.99615615, 0.08730106, 0.00717356], [0.08865273, -0.99601514, -0.00972295]])
    m_factor = th.zeros(190)+0.1#th.maximum(STD_spectra, th.tensor(0.1))
    # print(STD_spectra)
    # print(m_factor)
    sn_0 = 1320# or 750
    saving_folder = 'mygrad_1320'  # or mygrad_750

    with th.no_grad():
        loss_all = th.zeros((40, 190))
        loss_all_0 = th.zeros((40, 190))
        euc_dis = th.zeros((40, 190, 20))
        for sn in range(sn_0, sn_0 + 40):# #1[530,2070] #2[770,1910] #3[1110,1710] #4[1320,1560]
            print(sn)
            input_0, target_label = next(itertools.islice(training_generator_test, sn , None))#804 5712np.random.randint(5806)
            target_label_rel = absolute_relative(target_label).cuda()
            for k in range(0, 190):
                net_output_0, _ = model(input_0.cuda())
                loss_all_0[count, k] = loss(net_output_0, target_label_rel)
                input_1 = input_0.clone().detach()
                input_1[:, :, k] += m_factor[k]
                net_output, _ = model(input_1.cuda())
                loss_all[count, k] = loss(net_output, target_label_rel)
                euc_dis[count, k, :] = LA.norm(net_output_0.reshape(20, 3) - net_output.reshape(20, 3), dim=1)
            count += 1

        pathlib.Path(args.path_to_folder + '/results/' + saving_folder).mkdir(parents=True, exist_ok=True)
        # np.savetxt(args.path_to_folder + "/results/mygrad/loss_all_0.txt", loss_all_0.numpy().astype(float), fmt='%f')
        # np.savetxt(args.path_to_folder + "/results/mygrad/loss_all.txt", loss_all.numpy().astype(float), fmt='%f')
        th.save(loss_all_0, args.path_to_folder + '/results/' + saving_folder +'/loss_all_0.pt')
        th.save(loss_all, args.path_to_folder + '/results/' + saving_folder +'/loss_all.pt')
        # th.save(euc_dis, args.path_to_folder + '/results/' + saving_folder +'/euc_dis.pt')

        loss_all_0 = th.load(args.path_to_folder + 'results/' + saving_folder +'/loss_all_0.pt')
        loss_all = th.load(args.path_to_folder + 'results/' + saving_folder +'/loss_all.pt')
        euc_dis = th.load(args.path_to_folder + 'results/' + saving_folder +'/euc_dis.pt')
        input_0, target_label = next(itertools.islice(training_generator_test, sn_0, None))
        k = 189
        loss_all_0_mean = th.mean(loss_all_0, 0).cpu().numpy()
        # print(loss_all_0)
        # print(th.mean(loss_all, 0).shape)#190
        # print(STD_spectra.shape)#190
        # print(th.mean(euc_dis, 0).shape)#190,20
        loss_all_mean = (th.div((th.mean(loss_all, 0) - th.mean(loss_all_0, 0)), m_factor)).cpu().numpy()
        euc_dis_mean = th.transpose((th.div(th.transpose(th.mean(euc_dis, 0), 0, 1), m_factor)), 0, 1).cpu().numpy()

        # print(euc_dis_mean[20:35,:])
        # print(th.mean(euc_dis, 0)[20:35,:])
        # print(STD_spectra[0:60])

        # loss_all_mean = th.mean(loss_all, 0).cpu().numpy()
        # euc_dis_mean = th.mean(euc_dis, 0).cpu().numpy()

        # max_peaks, _ = find_peaks(input_0[:, 0, :].squeeze(), height=1)
        # print(max_peaks)
        plt.rcParams["figure.figsize"] = [25, 10]
        plt.rcParams["figure.autolayout"] = True
        # fig = plt.figure(figsize=(25, 10))
        # ax1 = plt.subplot()
        # # plt.subplot(1, 2, 1)
        # # plt.plot(wavelength, np.zeros(190) + loss_value_0.cpu().detach().numpy(), color='b',linewidth=0.5)
        # p0, = plt.plot(wavelength[np.arange(0, k + 1, 1)], loss_all_0_mean, color=Col[0], linewidth=1, label='Original Sample')
        # p1, = plt.plot(wavelength[np.arange(0, k + 1, 1)], loss_all_mean, color=Col[1], linestyle='None', marker='o', label='Modified Sample')
        # plt.ylabel('Smooth L1 Loss [a. u.]')
        # plt.xlabel('Wavelength [nm]')
        # # print(input.shape)[1,3,190]
        # ax2 = ax1.twinx()
        # p2, = ax2.plot(wavelength, input_0[:, 0, :].squeeze(), color=Col[2], linewidth=1, label='Spectrum Profile')
        # ax2.set_ylabel("Intensity [a. u.]")#, color="blue", fontsize=14
        # # plt.plot(wavelength[peaks], input_0[:, 0, peaks].squeeze(), "x")
        # for i in range(0, 15):
        #     p3, = plt.plot(np.zeros(2) + peaks[i], [-1, 3.5],"--", color="gray", linewidth=1, label='FBG Peaks (fitted)')#wavelength[peaks[i]]
        # # plt.plot(np.zeros_like(input_0[:, 0, :].squeeze()), "--", color="gray")
        #
        # plt.legend(handles=[p0, p1, p2, p3], loc='upper left')


        plt.subplot(2, 1, 1)
        colormap_0 = plt.cm.get_cmap('plasma')
        p2, = plt.plot(wavelength, input_0[:, 0, :].squeeze(), color="gray", linewidth=1, label='Spectrum Profile',alpha=0.3)
        loss_all_mean_norm = (loss_all_mean - np.min(loss_all_mean)) / np.max(loss_all_mean)
        # print(loss_all_mean_norm.shape)
        # print(loss_all_mean.shape)
        colors_0 = colormap_0(loss_all_mean_norm)
        plt.scatter(wavelength[np.arange(0, k + 1, 1)], input_0[:, 0, :].squeeze(), facecolor=colors_0,edgecolors='k', linestyle='-', marker='o',
                    label='Modified Sample', s=50)
        sm = plt.cm.ScalarMappable(cmap=colormap_0)
        sm.set_clim(vmin=0, vmax=1)
        p4 = plt.colorbar(sm)
        p4.ax.set_ylabel('Finite Difference [a. u.]')  # , rotation=270Smooth L1 Loss
        plt.ylabel('Intensity [a. u.]')
        plt.xlim([795, 895])
        # plt.xlabel('Wavelength [nm]')
        # # print(input.shape)[1,3,190]
        # ax2 = ax1.twinx()
        #
        # ax2.set_ylabel("Intensity [a. u.]")  # , color="blue", fontsize=14
        # # plt.plot(wavelength[peaks], input_0[:, 0, peaks].squeeze(), "x")
        # for i in range(0, 15):
        #     p3, = plt.plot(np.zeros(2) + peaks[i], [-1, 3.5], "--", color="gray", linewidth=1,
        #                    label='FBG Peaks (fitted)')  # wavelength[peaks[i]]
        # # plt.plot(np.zeros_like(input_0[:, 0, :].squeeze()), "--", color="gray")
        #
        # plt.legend(handles=[p0, p1, p2, p3], loc='upper left')
        #

        # plt.figure(figsize=(25, 10))
        plt.subplot(2, 1, 2)
        # ax3 = plt.subplot()

        euc_dis_norm = (euc_dis_mean-np.min(euc_dis_mean))/np.max(euc_dis_mean)
        colormap = plt.cm.get_cmap('viridis')
        for i in range(0, k+1):
            colors = colormap(euc_dis_norm[i, :])
            plt.scatter(np.zeros(20) + wavelength[i], np.arange(1, 21, 1), marker="s", facecolor=colors, s=80) #,alpha=0.7, linewidths=2
            sm = plt.cm.ScalarMappable(cmap=colormap)
            sm.set_clim(vmin=0, vmax=1)
            # plt.draw()
        # for i in range(0, 15):
        #     p5, = plt.plot(np.zeros(22) + peaks[i], np.arange(0, 22, 1)-1,"--", color="gray", label='FBG Peaks (fitted)')
        fbg = np.array([0.5, 3.5, 6.5, 9.5, 12.5])
        for i in range(0, 5):
            p6, = plt.plot(np.arange(795, 805, 1), np.zeros(10) + fbg[i],"--",  color=Col[6], label='Sensing Plane')

        p4 = plt.colorbar(sm)
        p4.ax.set_ylabel('Euclidean Distance [a. u.]')#, rotation=270

        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Marker')
        plt.ylim([-2, 23])
        plt.xlim([795, 895])
        # ax4 = ax3.twinx()
        # p7, = ax4.plot(wavelength, input_0[:, 0, :].squeeze(),".-", color="k", linewidth=1, label='Spectrum Profile')
        # ax4.set_ylabel("Normalized Intensity [a. u.]")
        # plt.legend(handles=[p6], loc='upper left')#p5,
        # plt.grid(which='minor', axis='x')
        plt.show()


if __name__ == "__main__":
    args = param.parser.parse_args()
    # dataset_for_inference = args.infer_data
    # checkpoint_to_load = args.infer_model

    pathlib.Path(args.path_to_folder+'/results/gradcam').mkdir(parents=True, exist_ok=True)

    dataset_for_inference = 'test'
    checkpoint_to_load = 'best'

    if dataset_for_inference == 'train':
        chose_path = args.traincsv_path
    elif dataset_for_inference == 'test':
        chose_path = args.testcsv_path
    elif dataset_for_inference == 'validate':
        chose_path = args.validatecsv_path
    else:
        raise NotImplementedError

    if checkpoint_to_load == 'last':
        path = './results/model/'+loading_from_folder+'/model_last.pt'
        # path = args.path_to_folder+'/results/model/model_last.pt'
        state = th.load(path)
        args = state['args']
        mod_model = 'last'
    elif checkpoint_to_load == 'best':
        path = './results/model/'+loading_from_folder+'/model_best.pt'
        # path = args.path_to_folder+'/results/model/model_best.pt'
        state = th.load(path)
        args = state['args']
        mod_model = 'best'
    else:
        raise NotImplementedError

    print("run Grad Cam on {} data, using the {} checkpoint".format(dataset_for_inference, checkpoint_to_load))
    print("change data using the --infer_data flag to train, validate or test")
    print("change checkpoint using the --infer_model flag to last or best")

    epoch_start = state['epoch']
    print("****************************************************************")
    print("************* epoch ", epoch_start, ", ***********************")
    print("************* path ", chose_path, ", ***********************")
    print("****************************************************************")

    print(chose_path)

    grad_cam(args, state, chose_path)
    # plt.show()


    # # Create a 2D tensor
    # T1 = th.Tensor([[3, 2], [7, 4], [6, 8]])
    # T1 = th.transpose(th.Tensor([[3, 2, 7], [7, 4, 6]]), 0, 1)
    # print(th.transpose(T1, 0, 1))
    #
    # # Create a 1-D tensor
    # T2 = th.Tensor([10, 2])
    # print("T1:\n", T1)
    # print("T2:\n", T2)
    #
    # # Divide 2-D tensor by 1-D tensor
    # v = th.div(T1, T2)
    # print("Element-wise division result:\n", v)