__author__ = "Carlo Seppi, Eva Schnider"
__copyright__ = "Copyright (C) 2019 Center for medical Image Analysis and Navigation"
__email__ = "carlo.seppi@unibas.ch"

import argparse

parser = argparse.ArgumentParser(description="Parameters for 1D Convolutional Nets")

parser.add_argument(
    "-m", action="store", dest="mode", default="restart", type=str, choices=["restart", "continue"],
    help="Choose whether to restart the training from scratch, or to continue from model_last."
)

parser.add_argument(
    "--infer-data", action="store", dest="infer_data", default="test", type=str, choices=["train", "validate", "test"],
    help="Choose which data partition to use for inference."
)

parser.add_argument(
    "--infer-model", action="store", dest="infer_model", default="last", type=str, choices=["last", "best"],
    help="Choose which data partition to use for inference."
)

parser.add_argument(
    "--logfile", action="store_true", dest="logfile", default=False,
    help="Add this flag if you want to redirect the console output to a file."
)


parser.add_argument(
    '--path-to-folder',
    default='./',
    help='name the folder'
)
#NaturePaper


parser.add_argument(
    '--path',
    default='./data/processed_data',
    help='path of the data'
)

parser.add_argument(
    '--traincsv-path', dest='traincsv_path',
    default='_train_Norm_Relative_1606Random_10-05-2022--15-12.csv',
    help='path of the csv train file'
)

parser.add_argument(
    '--testcsv-path', dest='testcsv_path',
   default='_test_Norm_Relative_1606Random_10-05-2022--15-12.csv',
   # default='_3min_Norm_10-05-2022_second.csv',
   # default='_3min_Norm_10-05-2022.csv',
   help='path of the csv test file'
)

parser.add_argument(
    '--validatecsv-path', dest='validatecsv_path',
    default='_val_Norm_Relative_1606Random_10-05-2022--15-12.csv',
    help='path of the csv Validate file'
)

parser.add_argument(
    '--classes',
    default=('dog', 'cat'),
    help='classes'
)

parser.add_argument(
    '--gpu-id', dest='gpu_id',
    type=int,
    default=0,
    help='gpu id if set to -1 then use cpu'
)

parser.add_argument(
    '--maxepoch',
    default=15000,
    help='Number of Epochs used in training'
)

parser.add_argument(
    '--nb-workers', dest='nb_workers',
    type=int,
    default=4,
    help='number of workers for the data loader'
)

parser.add_argument(
    '--amsgrad',
    default=True,
    type=lambda x: (str(x).lower()) == 'true',
    metavar='AM',
    help='Adam optimizer amsgrad parameter'
)

parser.add_argument(
    '--fcinputsize',
    default=[50, 15000],
    help='Min. and Max. number of Neurons allowed'
)

#########################################################################
