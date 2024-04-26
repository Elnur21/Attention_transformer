import torch
from torch import nn

import copy
import numpy as np

import matplotlib.pyplot as plt
from utils.utils import *

import os
import time
import math
import sys
import re

sys.path.append('../utils')
sys.path.append('..')

from utils.utils import read_all_datasets, prepare_data

# dataset_name = 'ArrowHead'
# dataset_name = 'MoteStrain'
datasets = [
    "LargeKitchenAppliances",
            # 'ArrowHead', 
            # 'BeetleFly', 
            # 'Ham', 
            # 'MoteStrain', 
            # 'OliveOil', 
            # 'Wine', 
            # 'Lightning7', 
            # 'InlineSkate', 
            # 'Beef', 
            # 'ACSF1', 
            # 'Yoga', 
            # 'GunPointOldVersusYoung',
            # 'FreezerSmallTrain', 
            # 'WordSynonyms', 
            # 'Car', 
            # 'ProximalPhalanxTW', 
            # 'InsectWingbeatSound',
            # 'FaceAll', 
            # 'EOGVerticalSignal',  
            # 'Earthquakes'
            ]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def test(model):
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


class Padding(nn.Module):

    def __init__(self, patch_size):
        super(Padding, self).__init__()
        self.patch_size = patch_size

    def forward(self, x):
        B, D, L= x.size()
        # print('x shape: ', x.shape)
        if L % self.patch_size != 0:
            padd_size = self.patch_size - L % self.patch_size
            # print('padd_size: ', padd_size)
            num_patches = math.ceil(L / self.patch_size)
            # print('num_patches: ', num_patches)
            last_elements = x[:, :, -1:]
            # print('last_elements: ', last_elements.shape)
            num_missing_values = self.patch_size - (L % self.patch_size)
            # print('num_missing_values: ', num_missing_values)
            if num_missing_values > 0:
                padding = last_elements.repeat(1, 1, num_missing_values)
                # print('padding shape: ', padding.shape)
                x = torch.cat([x, padding], dim=2)
                # print('x shape: ', x.shape)
            # print('final shape: ', x.shape)

            x = x.view(B, D, -1, self.patch_size)
        else:
            # print('B: ', B)
            # print('D: ', D)

            x = x.view(B, D, -1, self.patch_size)

        # print('before permutation: ', x.shape)
        return  x.permute(0, 1, 3, 2)
    

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class tAPE(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(tAPE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin((position * div_term)*(d_model/max_len))
        pe[:, 1::2] = torch.cos((position * div_term)*(d_model/max_len))
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        x = x + self.pe
        return self.dropout(x)




class Attention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout,n):
        super().__init__()
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5

        self.query = nn.Linear(emb_size, emb_size, bias=False)
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        # n = 20
        tmp = torch.arange(n) - torch.arange(n).unsqueeze(1)
        self.spec_attn_wghts = torch.abs(tmp) / n

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        attn = torch.matmul(q, k) * self.scale
        # print('attn: ', attn[0][0][0])
        # attn = nn.functional.softmax(attn, dim=-1)
        # print('attn: ', attn.shape)
        attn = torch.mul(attn, self.spec_attn_wghts.to(device))
        # print('special attn: ', tmp_.shape)
        attn = nn.functional.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2)
        out = out.reshape(batch_size, seq_len, -1)
        out = self.to_out(out)

        return out

class AttentionModel(nn.Module):
    def __init__(self, emb_size, num_heads, dropout, num_classes, dim_ff,num_features):
        super().__init__()

        print(dim_ff)
        # Define the max pooling layer
        self.pooling_layer = nn.AdaptiveAvgPool1d(int(dim_ff / 2))
        self.instance1d=nn.InstanceNorm1d(1, affine=True)
        self.revin_layer = RevIN(num_features)

        self.patching_layer = nn.Sequential(nn.Conv1d(1, emb_size, kernel_size=8, stride=4, padding='valid'),
                                         nn.BatchNorm1d(emb_size),
                                         nn.ReLU())
        
        emb_size = 32
        self.embed_layer = nn.Sequential(nn.Conv1d(8, emb_size, kernel_size=8, padding='same'),
                                         nn.BatchNorm1d(emb_size),
                                         nn.ReLU())
        
        # self.depthwise_conv = 
        emb_size = 16

        self.embed_layer2 = nn.Sequential( nn.Conv1d(
                                in_channels=32, 
                                out_channels=16, 
                                kernel_size=32, 
                                padding='same', 
                                groups=16, 
                                bias=False),

                                 nn.Conv1d(
                                in_channels=16, 
                                out_channels=16, 
                                kernel_size=1, 
                                bias=False
                                ),

                                nn.BatchNorm1d(emb_size),
                                nn.ReLU())
                                         
        self.Fix_pos_encode = tAPE(emb_size, max_len=dim_ff)
        self.Fix_pos_encode2 = tAPE(emb_size, max_len=dim_ff)
        # self.Fix_pos_encode3 = tAPE(emb_size, max_len=179)


        self.attn_layer = Attention(emb_size, num_heads, dropout,dim_ff)
        self.attn_layer2 = Attention(emb_size, num_heads, dropout,dim_ff)
        # self.attn_layer3 = Attention(emb_size, num_heads, dropout,dim_ff)


        self.LayerNorm1 = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)
        # self.LayerNorm3 = nn.LayerNorm(emb_size, eps=1e-5)


        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.to_out = nn.Linear(emb_size, num_classes)

        
    def forward(self, x):
        # x = self.instance1d(x)
        # print('input shape: ', x.shape)
        # x_in = self.revin_layer(x, 'norm')
        # x_out = self.revin_layer(x_in, 'denorm')
        # print('x_in shape: ', x_in.shape)
        # x_out = self.revin_layer(x_in, 'denorm')
        # print('x_out shape: ', x_out.shape)

        x = self.patching_layer(x)
        # print('Out patching shape: ', x.shape)
        x = x.squeeze(1)
        # print('Output squeeze: ', x.shape)
        
        x = self.embed_layer(x)
        # x = x.unsqueeze(1)
        x = self.embed_layer2(x)

        # x_src = self.embed_layer2(x).squeeze(2)
        # print('Output shape of embedding layer: ', x_src.shape)

        x_src = x.permute(0, 2, 1)
        x_src_pos = self.Fix_pos_encode(x_src)
        
        att = self.attn_layer(x_src_pos)
        # print('Output of the attention layer: ', x_src.shape)

        att = self.LayerNorm1(att + x_src)
        
        x_src_pos = self.Fix_pos_encode(att)
        att = self.attn_layer2(x_src_pos)

        att = self.LayerNorm2(att)
        # x_src_pos = self.Fix_pos_encode2(att)
        # att = self.attn_layer3(x_src_pos)



        # out = att + self.FeedForward(att)
        # att = self.LayerNorm2(att)
        out = att.permute(0, 2, 1)
        # out = x_src.permute(0, 2, 1)
        # out = att
        # print('After permutation: ', out.shape)
        # x_out = self.revin_layer(out, 'denorm')
        out = self.gap(out)
        # print('After gap: ', out.shape)
        out = self.flatten(out)
        # print('After flatten: ', out.shape)
        out = self.to_out(out)
        # print('classification layer ', out.shape)
        # print('\n\n')
        return out
    

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.01, adaptive=True, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


def run_code(dataset_name, trainloader, n,num_features):

    model = AttentionModel(8, 8, 0.2, 3, n,num_features)
    lossFn = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), )
    base_optimizer = torch.optim.Adam
    optimizer = SAM(model.parameters(), base_optimizer,)

    # optimizer = optim.Adam(model.parameters(), lr=0.001,)
    EPOCHS = 2000

    model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    min_train_loss = np.inf

    final_loss, learning_rates = [], []
    training_losses,training_accuracies = [], []
    val_losses, val_accuracies = [], []

    start_time = time.time()   
    for e in range(0, EPOCHS):
        print("===================================================================================")
        print('Epoch: ', e)


        model.train()

        # initlaize total training loss
        totalTainLoss = 0

        total = 0

        # initialize number of correct predictions in the training
        trainCorrect = 0

        for (x, y) in trainloader:
            optimizer.zero_grad()
            # send the input to the device
            x, y = x.to(device), y.to(device)
            # perform a forward pass and calculate the training loss
            pred = model(x.float())
            loss = lossFn(pred, y)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            lossFn(model(x.float()), y).backward()  
            optimizer.second_step(zero_grad=True)

            totalTainLoss += loss
            trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
            total += y.size(0)

        train_acc = 100. * trainCorrect / total
        # break
        print('Train acc: ', train_acc)
        print('Train loss: ', totalTainLoss.item())
        training_losses.append(totalTainLoss.item())
        training_accuracies.append(train_acc)
        test_loss, test_acc = test(model)
        print("lr: ",optimizer.param_groups[0]['lr'])

        if min_train_loss  >= totalTainLoss.item():
            min_train_loss = totalTainLoss.item()
            best_model_wts = copy.deepcopy(model.state_dict())

        val_losses.append(test_loss)
        val_accuracies.append(test_acc)
        final_loss.append(totalTainLoss.item())
        learning_rates.append(optimizer.param_groups[0]['lr'])
        print('\n')

    out_dir = './result_patch/'
    out_dir = out_dir + dataset_name + '/iter_' + str(iter) + '/'
    if os.path.exists(out_dir) == False:
        os.makedirs(out_dir)
    torch.save(best_model_wts, out_dir +  'best_model.pt')

    # training_losses_c = training_losses[0].cpu().detach().numpy()
    # Save Logs
    plot_loss_and_acc_curves(training_losses, val_losses, training_accuracies, val_accuracies, out_dir)

    duration = time.time() - start_time
    best_model = AttentionModel(8, 8, 0.2, 3, n,num_features)
    best_model.load_state_dict(best_model_wts)
    best_model.cuda()

    print('Best Model Accuracy in below ')
    start_test_time = time.time()
    test(best_model)
    test_duration = time.time() - start_test_time
    print(test(best_model))
    # for i in range(len(final_loss)):
    #     final_loss[i] = final_loss[i].cpu().detach().numpy()
    # final_loss = np.concatenate([np.atleast_1d(arr) for arr in final_loss])
    df = pd.DataFrame(list(zip(final_loss, learning_rates)), columns =['loss', 'learning_rate'])
    index_best_model = df['loss'].idxmin()
    row_best_model = df.loc[index_best_model]
    df_best_model = pd.DataFrame(list(zip([row_best_model['loss']], [index_best_model+1])), columns =['best_model_train_loss', 'best_model_nb_epoch'])

    df.to_csv(out_dir + 'history.csv', index=False)
    df_best_model.to_csv(out_dir + 'df_best_model.csv', index=False)

    loss_, acc_ = test(best_model)
    df_metrics = pd.DataFrame(list(zip([min_train_loss], [acc_], [duration], [test_duration])), columns =['Loss', 'Accuracy', 'Duration', 'Test Duration'])
    df_metrics.to_csv(out_dir + 'df_metrics.csv', index=False)

for dataset_name in datasets[:1]:
    print(dataset_name)
    datasets_dict = read_all_datasets([dataset_name])

    trainloader, valloader, input_shape, nb_classes = prepare_data(datasets_dict, dataset_name)
    num_features = len(datasets_dict[dataset_name][0][0])
    # try:
    n = (len(datasets_dict[dataset_name][0][0])-4)//4
    run_code(dataset_name,trainloader, n,num_features)
    # except Exception as e:
    #     print("e:",e)
    #     match = re.search(r'\d+', f"{e}")
    #     if match:
    #         first_number = int(match.group())
    #         print(first_number)
    #         run_code(dataset_name,trainloader, first_number,num_features)
    #     else:
    #         print("No number found in the text.")
