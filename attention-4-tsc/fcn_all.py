import torch
from torch import nn
import torch.nn.functional as F
from fastai import layers
# from .torch_core import *
from AbsolutePositionalEncodng import tAPE
from Attention import Attention
import math

NormType = layers.Enum('NormType', 'Batch BatchZero Weight Spectral')

class Padding(nn.Module):

    def __init__(self, patch_size):
        super(Padding, self).__init__()
        self.patch_size = patch_size

    def forward(self, x):
        B, D, L = x.size()
        padd_size = self.patch_size - L % self.patch_size
        
        num_patches = math.ceil(L / self.patch_size)
        last_elements = x[:, :, -1:]
        num_missing_values = self.patch_size - (L % self.patch_size)
        if num_missing_values > 0:
            padding = last_elements.repeat(1, 1, num_missing_values)
            x = torch.cat([x, padding], dim=2)

        return x

class SelfAttention(layers.Module):
    "Self attention layer for `n_channels`."
    def __init__(self, n_channels):
        self.query,self.key,self.value = [self._conv(n_channels, c) for c in (n_channels//8,n_channels//8,n_channels)]
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def _conv(self,n_in,n_out):
        return layers.ConvLayer(n_in, n_out, ks=1, norm_type=NormType.Spectral, ndim=1, act_cls=None, bias=False)

    def forward(self, x):
        #Notation from the paper.
        size = x.size()
        print('original x shape: ', x.shape)
        x = x.view(*size[:2],-1)
        print('reshaped x shape: ', x.shape)
        f,g,h = self.query(x),self.key(x),self.value(x)
        print('f shape: ', f.shape)
        print('f shape: ', f.transpose(1,2).shape)

        print('g shape: ', g.shape)

        beta = F.softmax(torch.bmm(g, f.transpose(1,2),))
        print('beta shape: ', beta.shape)
        # print('valu shape: ', h.shape)
        o = torch.bmm(h, beta) + x
        return o.view(*size).contiguous()

from typing import Optional
class Learned_Aggregation_Layer_multi(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.9,
        proj_drop: float = 0.95,
        num_classes: int = 3,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim: int = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # print('B. N and C: ', B, N, C)
        # print('shape ss: ', x[:, : self.num_classes].shape)
        # print('self.q(x[:, : self.num_classes]): ', self.q(x[:, : self.num_classes]).shape)
        # print('self.q(x[:, : self.num_classes]) reshape heads: ', 
        # self.q(x[:, : self.num_classes]).reshape(B, self.num_classes, self.num_heads, C // self.num_heads).shape)


        q = (
            self.q(x[:, : self.num_classes])
            .reshape(B, self.num_classes, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        # print('shape kk: ',x[:, self.num_classes:].shape)
        # print('self.k(x[:, self.num_classe:]): ', self.k(x[:, self.num_classes:]).shape)
        # print('self.k(x[:, self.num_classes: ]) reshape heads: ', 
        # self.k(x[:, self.num_classes:]).reshape(B, N - self.num_classes, self.num_heads, C // self.num_heads).shape)
        
        k = (
            self.k(x[:, self.num_classes:])
            .reshape(B, N - self.num_classes, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        q = q * self.scale
        v = (
            self.v(x[:, self.num_classes:])
            .reshape(B, N - self.num_classes, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        # print('query shape: ', q.shape)
        # print('key shape: ', k.shape)
        # print('value shape: ', v.shape)

        attn = q @ k.transpose(-2, -1)
        # print('attn shape: ', attn.shape)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, self.num_classes, C)

        # print('attn @ v shape: ', (attn @ v).shape)
        # print('attn @ v transpose(1, 2) shape: ', (attn @ v).transpose(1, 2).shape)
        # print('x_cls shape: ', x_cls.shape)
        # x_cls = self.proj(x_cls)
        # x_cls = self.proj_drop(x_cls)
        # print('x_cls shape at the end: ', x_cls.shape)

        # print('\n')
        
        return x_cls


class Learned_Aggregation_Layer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.4,
        proj_drop: float = 0.4,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim: int = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.id = nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = q @ k.transpose(-2, -1)
        attn = self.id(attn)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls



import torch
import torch.nn as nn
from typing import Optional

class LearnedAggregationLayerMultiPatch(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.3,
        proj_drop: float = 0.4,
        num_patches: int = 26,  # Number of patches
        num_classes: int = 3,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, num_classes)
        self.proj_drop = nn.Dropout(proj_drop)
        self.num_classes = num_classes
        self.num_patches = num_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        # Calculate the patch size
        patch_size = N // self.num_patches
        
        # Reshape input tensor into patches
        x = x.view(B, self.num_patches, patch_size, C)

        # Linear transformation for Query, Key, and Value
        q = self.q(x).reshape(B, self.num_patches, patch_size, self.num_heads, -1).permute(0, 3, 1, 4, 2) # B x num_heads x num_patches x head_dim x patch_size
        k = self.k(x).reshape(B, self.num_patches, patch_size, self.num_heads, -1).permute(0, 3, 1, 4, 2)
        v = self.v(x).reshape(B, self.num_patches, patch_size, self.num_heads, -1).permute(0, 3, 1, 4, 2)

        q = q * self.scale

        attn = (q @ k.transpose(-1, -2))  # B x num_heads x num_patches x patch_size x patch_size
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).permute(0, 2, 3, 1, 4).reshape(B, self.num_patches, patch_size, -1) # B x num_patches x patch_size x num_heads*head_dim
        x_cls = x_cls.mean(dim=-2) # Average pooling over the heads
        x_cls = x_cls.reshape(B, self.num_patches, -1) # B x num_patches x num_heads*head_dim
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls

class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.2,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class Classifier_FCN(nn.Module):

    def __init__(self, input_shape, nb_classes, filter_count):
        super(Classifier_FCN, self).__init__()
        self.nb_classes = nb_classes
        self.filter_count = filter_count

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=self.filter_count, kernel_size=8, stride=1, padding='same', bias=False)
        self.bn1 = nn.BatchNorm1d(self.filter_count,)
        self.relu = nn.ReLU()
        # self.feat_fc_1 = nn.Linear(self.filter_count, self.nb_classes)
        
        self.conv2 = nn.Conv1d(in_channels=self.filter_count, out_channels=self.filter_count*2, kernel_size=5, stride=1, padding='same', bias=False)
        self.bn2 = nn.BatchNorm1d(self.filter_count*2)
        # self.feat_fc_2 = nn.Linear(self.filter_count*2, self.nb_classes)

        self.conv3 = nn.Conv1d(in_channels=self.filter_count*2, out_channels=self.filter_count, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn3 = nn.BatchNorm1d(self.filter_count)
        # self.feat_fc_3 = nn.Linear(self.filter_count, self.nb_classes)    

        # self.cus_gap = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=251,  padding='valid')

        # self.cus_gap_2 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=128, kernel_size=251,  padding='valid'),
        #                                 nn.BatchNorm1d(128),
        #                                 # nn.GELU()
        #                                 )

        # self.embed_layer = nn.Sequential(nn.Conv2d(1, 1, kernel_size=[1, 251], padding='valid'),
        #                                  nn.BatchNorm2d(128),
        #                                  nn.ReLU())

        # self.embed_layer2 = nn.Sequential(nn.Conv2d(64, 16, kernel_size=[128, 1], padding='valid'),
        #                                   nn.BatchNorm2d(16),
        #                                   nn.ReLU())
        # self.cus_gap = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=251,  padding='valid')

        # self.cus_gap_2 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=128, kernel_size=251,  padding='valid'),
        #                                 nn.BatchNorm1d(128),
        #                                 # nn.ReLU()
        #                                 )

        # self.embed_layer_1 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=64, stride=1, kernel_size=20, padding='same', bias=False),
        #                                  nn.BatchNorm1d(64),
        #                                  nn.ReLU())

        # self.embed_layer_2 = nn.Sequential(nn.Conv1d(64, 32, kernel_size=10, padding='same'),
        #                                   nn.BatchNorm1d(32),
        #                                   nn.ReLU())

        self.Fix_Position = tAPE(128, dropout=0.01, max_len=251)
        self.padding = Padding(10)
        # self.attn_agg_layer = Learned_Aggregation_Layer(128, )
        self.attn_agg_layer_multi = Learned_Aggregation_Layer_multi(128, num_classes=3)
        self.simple_mlp = MLP(128, 64, 32)

        # self.attn_layer = Attention(128, 8, 0.8)
        


        # self.attn_drop = nn.Dropout(0.1)
        # self.LayerNorm1 = nn.LayerNorm(128, eps=1e-5)
        # self.LayerNorm2 = nn.LayerNorm(128, eps=1e-5)
        # self.FeedForward = nn.Sequential(
            # nn.Linear(128, 256),
            # nn.ReLU(),
            # nn.Dropout(0.1),
            # nn.Linear(256, 128),
            # nn.Dropout(0.1))

        # self.relu = nn.ReLU()
        self.avgpool1 = nn.AdaptiveAvgPool1d(1)
        # self.attn_pool = SelfAttention(128)

        self.fc1 = nn.Linear(96, self.nb_classes)

    def forward(self, x):
        # First convolutional block
        # print('Input shape: ', x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print('Input shaoe of first conv block: ', x.shape)

        # Second convolutional block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # print('Input shaoe of second conv block: ', x.shape)

        # Third convolutional block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        # print('Input shaoe of third conv block: ', x.shape)

        # print('Input shape: ', x.shape)
        # x = self.cus_gap_2(x)
        # num_positive_values = (x > 0).sum(dim=-1)  # Sum along the last dimension
        # total_elements = x.size(-1)
        # x = num_positive_values / total_elements

        # print('Output of custom gap shape: ', x.shape)
        # print('Input shape before embedding layer: ', x.shape)
        # x = self.embed_layer_1(x)
        # print('Input shape after first embedding layer: ', x.shape)
        # x = self.embed_layer_2(x)
        # print('Input shape after second embedding layer: ', x.shape)

        # x = self.padding(x)
        # print('After padding the input shape: ', x.shape)
        # print('the input shape: ', x_src.shape)
        
        x_src_pos = self.Fix_Position(x)
        # print('Output shape of Pos Enc: ', x_src_pos.shape)

        x = self.attn_agg_layer_multi(x_src_pos)
        # print('attention out shapr: ', att.shape)
        # print('MLP input shape: ', x.shape)
        x = x.permute(0, 2, 1)

        # x = self.simple_mlp(x)
        # print('MLP output shape: ', x.shape)

        # print('Out x shape: ', x.shape)
        # att = self.LayerNorm1(att)
        # out = att + self.FeedForward(att)
        # out = self.LayerNorm2(out)

        # att = self.LayerNorm(x)
        # out = att + self.FeedForward(att)
        # out = self.LayerNorm2(out)

        # print('Input shape to Pos Enc: ', x.shape)
        # print('Input shape of Attention layer: ', x.shape)

        # x = self.attn_agg_layer(x)
        # print('Output of Attention Aggregation layer: ', x.shape)
        # x = x.permute(0, 2, 1)
        # print('Output shape of Aggregation layer after permutation: ', x.shape)
        # x = self.attn_pool(x)
        # x = x.unsqueeze(1)
        # print('Input shape before embedding layer: ', x.shape)
        # x = self.embed_layer(x)
        # print('Output of first conv2d block: ', x.shape)
        # x = self.embed_layer2(x)
        # print('Output of second conv2d block: ', x.shape)
        # x = x.squeeze(2)
        

        # print('Output shape: ', x.shape)
        x = x.permute(0, 2, 1)

        # x = self.avgpool1(x)
        x = x.reshape(x.shape[0], -1)
        # print('tttt last out shape: ', x.shape)
        x = self.fc1(x)
        # print('\n')
        return x

    def get_bn_before_relu(self):
        bn1 = self.bn1
        bn2 = self.bn2
        bn3 = self.bn3

        return [bn1, bn2, bn3]


    def extract_features(self, x):
        features = []
        # First convolutional block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) 
        feat1 = F.adaptive_avg_pool1d(x, 1)
        feat1 = feat1.reshape(feat1.shape[0], -1)
        feat1 = self.feat_fc_1(feat1)
        feat_dist_1 = F.log_softmax(feat1, dim=1)
        features.append(feat_dist_1)
        
        # feat1 = F.adaptive_avg_pool1d(x, 1)
        # feat1 = feat1.reshape(feat1.shape[0], -1)

        # Second convolutional block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        feat2 = F.adaptive_avg_pool1d(x, 1)
        feat2 = feat2.reshape(feat2.shape[0], -1)
        feat2 = self.feat_fc_2(feat2)
        feat_dist_2 = F.log_softmax(feat2, dim=1)
        features.append(feat_dist_2)

        # # # feat2 = F.adaptive_avg_pool1d(x, 1)
        # # # feat2 = feat2.reshape(feat2.shape[0], -1)

        # Third convolutional block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        feat3 = F.adaptive_avg_pool1d(x, 1)
        feat3 = feat3.reshape(feat3.shape[0], -1)
        feat3 = self.feat_fc_3(feat3)
        feat_dist_3 = F.log_softmax(feat3, dim=1)
        features.append(feat_dist_3)
        
        # feat3 = F.adaptive_avg_pool1d(x, 1)
        # feat3 = feat3.reshape(feat3.shape[0], -1)


        x = self.avgpool1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)


        gamma = self.bn3.weight.data.clone().cpu().detach().numpy()
        beta = self.bn3.bias.data.clone().cpu().detach().numpy()
        

        return features, x, gamma, beta