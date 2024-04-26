import torch
from torch import nn
from classifiers.AbsolutePositionalEncoding import tAPE as AbsolutePositionalEncoding 
import torch
import torch.nn as nn
# from einops import rearrange
import pandas as pd
# from einops.layers.torch import Rearrange
from torch import Tensor
import math


class Padding(nn.Module):

    def __init__(self, patch_size):
        super(Padding, self).__init__()
        self.patch_size = patch_size

    def forward(self, x):
        print(x.size())
        B, L = x.size()
        padd_size = self.patch_size - L % self.patch_size
        
        num_patches = math.ceil(L / self.patch_size)
        last_elements = x[:, :, -1:]
        num_missing_values = self.patch_size - (L % self.patch_size)
        if num_missing_values > 0:
            padding = last_elements.repeat(1, 1, num_missing_values)
            x = torch.cat([x, padding], dim=2)

        return x


# class PatchEmbedding(nn.Module):
#     def __init__(self, in_channels=128, patch_size=10):
#         self.patch_size = patch_size
#         super().__init__()
#         self.projection = nn.Sequential(
#             Rearrange('b c (h p) -> b h (p c)', p=patch_size,),
#         )

#     def forward(self, x:Tensor) -> Tensor:
#         x = self.projection(x)
#         return x

class Attention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5

        self.query = nn.Linear(emb_size, emb_size, bias=False)
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        attn = torch.matmul(q, k) * self.scale
        attn = nn.functional.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2)
        out = out.reshape(batch_size, seq_len, -1)
        out = self.to_out(out)

        return out

class AttentionModel(nn.Module):
    def __init__(self, emb_size, num_heads, dropout, num_classes, dim_ff):
        super().__init__()
        self.embed_layer = nn.Sequential(nn.Conv1d(1, emb_size, kernel_size=8, padding='same'),
                                         nn.BatchNorm1d(emb_size),
                                         nn.GELU())

        self.embed_layer2 = nn.Sequential(nn.Conv2d(1, emb_size, kernel_size=[8, 1], padding='valid'),
                                    nn.BatchNorm2d(emb_size),
                                    nn.GELU())
                                         
        self.Fix_pos_encode = AbsolutePositionalEncoding(emb_size, max_len=300)
        self.attn_layer = Attention(emb_size, num_heads, dropout)

        self.LayerNorm1 = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(dropout)
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.to_out = nn.Linear(emb_size, num_classes)

        
    def forward(self, x):
        print('Input shape: ', x.shape)
        patching_layer = Padding(patch_size=8)
        x = patching_layer(x)
        print('Output shape of patching layer: ', x.shape)
        x_src = self.embed_layer2(x).squeeze(2)
        print('Output shape of embedding layer: ', x_src.shape)

        x_src = x_src.permute(0, 2, 1)
        print('Input to the attention layer: ', x_src.shape)
        # x_src = self.Fix_pos_encode(x_src)
        x = x_src + self.attn_layer(x_src)
        # att = self.LayerNorm1(x)
        # out = att + self.FeedForward(att)
        # out = self.LayerNorm2(out)
        out = x.permute(0, 2, 1)
        # out = x_src.permute(0, 2, 1)
        # print('After permutation: ', x_src.shape)
        out = self.gap(out)
        # print('After gap: ', out.shape)
        out = self.flatten(out)
        # print('After flatten: ', out.shape)
        out = self.to_out(out)
        # print('classification layer ', out.shape)
        return out