import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch


def read_all_datasets(DATASET_NAMES_2018) -> pd.DataFrame:
    cur_root_dir = '/home/elnur/Desktop/new/LITE_CoTrain/UCRArchive_2018'
    datasets_dict = {}

    for dataset_name in DATASET_NAMES_2018:
        df_train =  pd.read_csv(cur_root_dir + '/' + dataset_name + '/' + dataset_name + '_TRAIN.tsv', sep='\t', header=None)
        df_test =  pd.read_csv(cur_root_dir + '/' + dataset_name + '/' + dataset_name + '_TEST.tsv', sep='\t', header=None)

        y_train = df_train.values[:, 0]
        y_test = df_test.values[:, 0]
        
        x_train = df_train.drop(columns=[0])
        x_test = df_test.drop(columns=[0])
        
        x_train.columns = range(x_train.shape[1])
        x_test.columns = range(x_test.shape[1])
        
        x_train = x_train.values
        x_test = x_test.values
        
        # znorm
        std_ = x_train.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_
        
        std_ = x_test.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_
        
        datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(), y_test.copy())


    return datasets_dict

def prepare_data(datasets_dict, dataset_name):
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # make the min to zero of labels
    y_train, y_test = transform_labels(y_train, y_test)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    new_x_train = np.transpose(x_train, (0, 2, 1))
    new_x_test = np.transpose(x_test, (0, 2, 1))
    
    x_train, y_train= torch.from_numpy(new_x_train), torch.from_numpy(y_train)
    x_test, y_test= torch.from_numpy(new_x_test), torch.from_numpy(y_test)
    input_shape = x_train.shape[1:]

    mini_batch_size = 16
    # def seed_worker(worker_id):
    #     worker_seed = torch.initial_seed() % 2**32
    #     np.random.seed(worker_seed)
    torch.manual_seed(42)

    trainloader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=mini_batch_size,
        shuffle=False,
        # worker_init_fn=seed_worker,
    )

    valloader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=mini_batch_size,
        shuffle=False
    )

    print('batch_size is ', mini_batch_size)
    return trainloader, valloader, input_shape, nb_classes



def znorm(x, dim=2):

    stds = np.std(x, axis=dim, keepdims=True)
    if len(stds[stds == 0.0]) > 0:
        stds[stds == 0.0] = 1.0
        return (x - x.mean(axis=dim, keepdims=True)) / stds
    
    return (x - x.mean(axis=dim, keepdims=True)) / (x.std(axis=dim, keepdims=True))


def encode_labels(y):

    labenc = LabelEncoder()

    return labenc.fit_transform(y)



def transform_labels(y_train, y_test):
    """
    Transform label to min equal zero and continuous
    For example if we have [1,3,4] --->  [0,1,2]
    """
    # no validation split
    # init the encoder
    encoder = LabelEncoder()
    # concat train and test to fit
    y_train_test = np.concatenate((y_train, y_test), axis=0)
    # fit the encoder
    encoder.fit(y_train_test)
    # transform to min zero and continuous labels
    new_y_train_test = encoder.transform(y_train_test)
    # resplit the train and test
    new_y_train = new_y_train_test[0:len(y_train)]
    new_y_test = new_y_train_test[len(y_train):]
    return new_y_train, new_y_test

def plot_loss_and_acc_curves(training_losses, val_losses, training_accuracies, val_accuracies, out_dir):
    plt.plot(training_losses, label='train_loss')
    plt.plot(val_losses, label='val_loss')
    plt.savefig(out_dir + 'losses.png')
    plt.clf()
    plt.cla()
    plt.close()
    plt.plot(training_accuracies, label='train_acc')
    plt.plot(val_accuracies, label='val_acc')
    plt.savefig(out_dir + 'accuracies.png')
    plt.clf()
    plt.cla()
    plt.close()


def plot_feature_and_ce_losses(ce_losses, final_feat_loss_list, out_dir):
    plt.plot(ce_losses, label='CE loss')
    plt.plot(final_feat_loss_list, label='Feature loss')
    plt.legend()
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss values')
    plt.title('CE and Feature loss (MSE) across epochs')
    plt.savefig(out_dir + 'losses_2.png')



def random_orthogonal_init(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            # Initialize the weights as random orthogonal matrices
            if param.ndimension() >= 2:
                weight = param.data
                torch.nn.init.orthogonal_(weight)

import numpy as np

def coefficients_function(epoch, period=100):
    """
    Function to generate coefficients for loss functions based on sine and cosine functions.
    
    Parameters:
        epoch (int): The current epoch number.
        period (int): Period of the sine and cosine functions. Default is 100 epochs.
        
    Returns:
        tuple: A tuple containing the coefficients for the two loss functions.
    """
    # Calculate the angle based on the current epoch and the period
    angle = 2 * np.pi * epoch / period

    # Calculate coefficients using sine and cosine functions
    coefficient1 = np.sin(angle)**2
    coefficient2 = 1 - coefficient1

    return coefficient1, coefficient2

