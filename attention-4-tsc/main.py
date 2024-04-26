import copy
import time
import numpy as np
import pandas as pd
import os

from HELPER import SAM

import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
import torch.backends.cudnn as cudnn
import torch.optim as optim
from Attention import AttentionModel
# from fcn_copy import Classifier_FCN
from utils.utils import *
from utils.constants import UNIVARIATE_DATASET_NAMES_2018

def fit_classifier(trainloader, valloader, input_shape, nb_classes, dataset_name, epochs, iter, out_dir, filter_count):
    # For reproducibility
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # cudnn.deterministic = True
    # cudnn.benchmark = False
    # print('seed: ', seed)
    # for randomness
    cudnn.benchmark = True

    # model = Attention(input_shape, nb_classes, filter_count)
    model = AttentionModel(16, 8, 0.2, 3, 64)
    lossFn = nn.CrossEntropyLoss()
    base_optimizer = torch.optim.Adam
    optimizer = SAM(model.parameters(), base_optimizer,)
    # optimizer = optim.Adam(model.parameters(), lr=0.001,)
    print('model created ', )
    use_cuda = torch.cuda.is_available()  
    print('use_cuda: ', use_cuda)              
    if use_cuda:
        print('Inside')
        torch.cuda.set_device(0)
        model.cuda()
        # summary(model, (input_shape[-2], input_shape[-1]))


    # Training
    def train_alone_model(net, epoch):
        print('\n\n Training epoch: %d' % epoch)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net.train()

        train_loss = 0
        correct = 0
        total = 0

        
        for batch_idx, (inputs, targets) in enumerate(trainloader):

            
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = net(inputs.float())
            loss = lossFn(outputs, targets)
            loss.backward() 

        
            optimizer.first_step(zero_grad=True)

            lossFn(model(inputs.float()), targets).backward()  
            optimizer.second_step(zero_grad=True)  

            train_loss += loss

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().float().item()

            b_idx = batch_idx

        print('Training Loss: %.3f | Training Acc: %.3f%% (%d/%d)' % (train_loss / (b_idx + 1), 100. * correct / total, correct, total))

        return train_loss / (b_idx + 1), 100. * correct / total

    def test(model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            model.eval()
            test_loss = 0
            correct = 0
            total = 0

            for b_idx, (x, y) in enumerate(valloader):
                (x, y) = x.to(device), y.to(device)
                pred = model(x.float())

                lossFn = nn.CrossEntropyLoss()
                loss = lossFn(pred, y)
                test_loss += loss.item()

                total += y.size(0)
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            print('Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)' % (test_loss / (b_idx + 1), 100. * correct / total, correct, total))
            return test_loss / (b_idx + 1), 100. * correct / total

    
    # optimizer = optim.Adam(model.parameters(),)
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 1e-3, epochs=epochs, steps_per_epoch=len(trainloader))
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True, min_lr=0.0001)
    # optimizer=torch.optim.Adadelta(model.parameters(), lr=1e-1, eps=1e-8)
    # optimizer = optim.Adam(model.parameters(), lr=0.001,)
    base_optimizer = torch.optim.Adam
    optimizer = SAM(model.parameters(), base_optimizer,)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 1e-3, epochs=epochs, steps_per_epoch=len(trainloader))

    use_cuda = torch.cuda.is_available()        
    best_model_wts = copy.deepcopy(model.state_dict())
    min_train_loss = np.inf
    
    final_loss, learning_rates = [], []
    training_losses,training_accuracies = [], []
    val_losses, val_accuracies = [], []

    start_time = time.time()    
    for epoch in range(epochs):
        train_loss, train_acc = train_alone_model(model, epoch)
        
        training_losses.append(train_loss)
        training_accuracies.append(train_acc)
        val_loss, val_acc = test(model)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        if min_train_loss  >= train_loss:
            min_train_loss = train_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        scheduler.step()

        final_loss.append(train_loss)
        learning_rates.append(optimizer.param_groups[0]['lr'])
    
    out_dir = out_dir + dataset_name + '/iter_' + str(iter) + '/'
    if os.path.exists(out_dir) == False:
        os.makedirs(out_dir)
    torch.save(best_model_wts, out_dir +  'best_model.pt')

    training_losses_c = training_losses[0].cpu().detach().numpy()
    # Save Logs
    plot_loss_and_acc_curves(training_losses_c, val_losses, training_accuracies, val_accuracies, out_dir)

    duration = time.time() - start_time
    best_model = AttentionModel(16, 8, 0.2, 3, 64)
    best_model.load_state_dict(best_model_wts)
    best_model.cuda()
    
    print('Best Model Accuracy in below ')
    start_test_time = time.time()
    test(best_model)
    test_duration = time.time() - start_test_time
    print(test(best_model))
    for i in range(len(final_loss)):
        final_loss[i] = final_loss[i].cpu().detach().numpy()
    final_loss = np.concatenate([np.atleast_1d(arr) for arr in final_loss])
    print(final_loss)
    df = pd.DataFrame(list(zip(final_loss, learning_rates)), columns =['loss', 'learning_rate'])
    index_best_model = df['loss'].idxmin()
    row_best_model = df.loc[index_best_model]
    df_best_model = pd.DataFrame(list(zip([row_best_model['loss']], [index_best_model+1])), columns =['best_model_train_loss', 'best_model_nb_epoch'])

    df.to_csv(out_dir + 'history.csv', index=False)
    df_best_model.to_csv(out_dir + 'df_best_model.csv', index=False)

    loss_, acc_ = test(best_model)
    df_metrics = pd.DataFrame(list(zip([min_train_loss], [acc_], [duration], [test_duration])), columns =['Loss', 'Accuracy', 'Duration', 'Test Duration'])
    df_metrics.to_csv(out_dir + 'df_metrics.csv', index=False)



# Main function
UNIVARIATE_DATASET_NAMES_2018 = ["LargeKitchenAppliances"]
datasets_dict = read_all_datasets(UNIVARIATE_DATASET_NAMES_2018)

filter_count = 128
epochs = 1
num_runs = 1
out_dir = './result_patch/'

for dataset_name in UNIVARIATE_DATASET_NAMES_2018:
    for i in range(num_runs):
        trainloader, valloader, input_shape, nb_classes = prepare_data(datasets_dict, dataset_name)
        fit_classifier(trainloader, valloader, input_shape, nb_classes, dataset_name, epochs, i+1, out_dir, filter_count)        