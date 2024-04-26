import torch
import torch.nn as nn
# from einops import rearrange
import pandas as pd
from einops.layers.torch import Rearrange
from torch import Tensor
import math


class tAPE(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence. 
        The positional encodings have the same dimension as the embeddings, so that the two can be summed. 
        Here, we use sine and cosine functions of different frequencies.
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

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
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
        # print('Input shape ', x.shape)
        # print('self.pe shape ', self.pe.shape)

        x = x + self.pe
        return self.dropout(x)



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


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=128, patch_size=10):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            Rearrange('b c (h p) -> b h (p c)', p=patch_size,),
        )

    def forward(self, x:Tensor) -> Tensor:
        x = self.projection(x)
        return x



class Attention_Patch(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads

        self.scale = emb_size ** -0.5
        self.num_patches =  26
        self.patch_size = 10
        self.padding = Padding(10)
        self.patching = PatchEmbedding()
        

        # self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        # self.to_out = nn.LayerNorm(emb_size)
        # self.qkv = nn.Linear(emb_size, emb_size * 3)
        # self.fc = nn.Linear(emb_size, emb_size)
        # self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        # print('Attention input shape: ', x.shape)

        batch_size, seq_len, C = x.shape
        
        # print('Original input shape: ', x.shape)
        x = self.patching(x)
        # print('After patch embedding: ', x.shape)

        # x = x.view(batch_size, self.num_patches, self.patch_size, C)
        # print('After reshaping input shape: ', x.shape)



        # Linear transformation for Query, Key, and Value
        # qkv = self.qkv(x).chunk(3, dim=-1)  # Split into Query, Key, and Value
        # q, k, v = [x.reshape(batch_size, self.num_patches, -1, self.num_heads, self.head_dim).transpose(2, 3) for x in qkv]

        # Compute scaled dot-product attention
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        # attn = self.dropout(attn)

        # Apply attention to the values
        # x = (attn @ v).transpose(2, 3).reshape(batch_size, self.num_patches, -1, C)

        # Reshape back to original shape
        x = x.view(batch_size, seq_len, C)

        # Final linear transformation
        x = self.fc(x)










        # k1 = self.key(x)
        # print('key shape: ', k1.shape)
        # k2 = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1)
        # print('key shape after reshaping: ', k2.shape)

        # q1 = self.query(x)
        # print('query shape: ', q1.shape)
        # q2 = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1)
        # print('query shape after reshaping: ', q2.shape)

        # v1 = self.value(x)
        # print('value shape: ', v1.shape)
        # v2 = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1)
        # print('value shape after reshaping: ', v2.shape)

        # k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        # v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)



        # k,v,q shape = (batch_size, num_heads, seq_len, d_head)

        # print('Last key shape: ', k.shape)
        # print('Last value shape: ', v.shape)
        # print('Last query shape: ', q.shape)
        # attnn = torch.matmul(q, k)
        # print('Attention shape: ', attnn.shape)
        # attn = torch.matmul(q, k) * self.scale
        # attn shape (seq_len, seq_len)
        # attn = nn.functional.softmax(attn, dim=-1)
        # print('Last attn shape: ', attn.shape)
        # print(attn[0][0][0])
        # import matplotlib.pyplot as plt
        # plt.plot(x[0, :, 0].detach().cpu().numpy())
        # plt.show()

        # out = torch.matmul(attn, v)
        # print('out 1 shape: ', out.shape)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        # out = out.transpose(1, 2)
        # print('out 2 shape: ', out.shape)

        # out.shape == (batch_size, seq_len, num_heads, d_head)
        # out = out.reshape(batch_size, seq_len, -1)
        # print('out 3 shape: ', out.shape)

        # out.shape == (batch_size, seq_len, d_model)
        # out = self.to_out(out)
        return x


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
        self.embed_layer = nn.Sequential(nn.Conv1d(1, emb_size, kernel_size=8, stride=4, padding='valid'),
                                         nn.BatchNorm1d(emb_size),
                                         nn.ReLU())

        # self.embed_layer2 = nn.Sequential(nn.Conv2d(1, emb_size, kernel_size=[8, 1], padding='valid'),
        #                             nn.BatchNorm2d(emb_size),
        #                             nn.GELU())
                                         
        self.Fix_pos_encode = tAPE(emb_size, max_len=179)
        # self.Fix_pos_encode2 = tAPE(emb_size, max_len=179)
        # self.Fix_pos_encode3 = tAPE(emb_size, max_len=179)


        self.attn_layer = Attention(emb_size, num_heads, dropout)
        # self.attn_layer2 = Attention(emb_size, num_heads, dropout)
        # self.attn_layer3 = Attention(emb_size, num_heads, dropout)


        self.LayerNorm1 = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)
        # self.LayerNorm3 = nn.LayerNorm(emb_size, eps=1e-5)


        # self.FeedForward = nn.Sequential(
        #     nn.Linear(emb_size, dim_ff),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(dim_ff, emb_size),
        #     nn.Dropout(dropout)
        # )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.to_out = nn.Linear(emb_size, num_classes)

        
    def forward(self, x):
        # print('Input shape: ', x.shape)
        # patching_layer = Padding(patch_size=8)
        # x = patching_layer(x)
        x = self.embed_layer(x)
        print(x.shape)
        x = x.squeeze(1)

        # print('Output shape of embedding layer: ', x.shape)
        # x_src = self.embed_layer(x)
        # print('embedding shape: ', x_src.shape)
        # x_src = self.embed_layer2(x).squeeze(2)
        # print('Output shape of embedding layer: ', x_src.shape)

        x_src = x.permute(0, 2, 1)
        # print('Input to the attention layer: ', x_src.shape)
        x_src_pos = self.Fix_pos_encode(x_src)
        att = self.attn_layer(x_src_pos)
        # att = self.LayerNorm1(att + x_src)
        
        # x_src_pos = self.Fix_pos_encode(att)
        # att = self.attn_layer2(x_src_pos)

        att = self.LayerNorm2(att)
        # x_src_pos = self.Fix_pos_encode3(att)
        # att = self.attn_layer3(x_src_pos)


        # out = att + self.FeedForward(att)
        # out = self.LayerNorm2(out)
        out = att.permute(0, 2, 1)
        # out = x_src.permute(0, 2, 1)
        # print('After permutation: ', x_src.shape)
        out = self.gap(out)
        # print('After gap: ', out.shape)
        out = self.flatten(out)
        # print('After flatten: ', out.shape)
        out = self.to_out(out)
        # print('classification layer ', out.shape)
        return out

