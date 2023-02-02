#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 12:13:06 2023

@author: ravi
"""

from __future__ import print_function
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y


class GLSTM(nn.Module):
    def __init__(
        self,
        in_features=None,
        out_features=None,
        mid_features=None,
        hidden_size=448,
        groups=2,
    ):
        super(GLSTM, self).__init__()

        hidden_size_t = hidden_size // groups

        self.lstm_list1 = nn.ModuleList(
            [
                nn.LSTM(hidden_size_t, hidden_size_t, 1, batch_first=True)
                for i in range(groups)
            ]
        )
        self.lstm_list2 = nn.ModuleList(
            [
                nn.LSTM(hidden_size_t, hidden_size_t, 1, batch_first=True)
                for i in range(groups)
            ]
        )

        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.groups = groups
        self.mid_features = mid_features

    def forward(self, x):
        out = x
        out = out.transpose(1, 2).contiguous()
        out = out.view(out.size(0), out.size(1), -1).contiguous()

        out = torch.chunk(out, self.groups, dim=-1)
        out = torch.stack(
            [self.lstm_list1[i](out[i])[0] for i in range(self.groups)], dim=-1
        )
        out = torch.flatten(out, start_dim=-2, end_dim=-1)
        out = self.ln1(out)

        out = torch.chunk(out, self.groups, dim=-1)
        out = torch.cat(
            [self.lstm_list2[i](out[i])[0] for i in range(self.groups)], dim=-1
        )
        out = self.ln2(out)

        out = out.view(out.size(0), out.size(1), x.size(1), -1).contiguous()
        out = out.transpose(1, 2).contiguous()

        return out


class GluConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(GluConv2d, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.sigmoid(self.conv2(x))
        out = out1 * out2
        return out


class GluConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(GluConv1d, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            padding_mode="replicate",
        )
        self.conv2 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            padding_mode="replicate",
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.sigmoid(self.conv2(x))
        out = out1 * out2
        return out


class GluConvTranspose2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, output_padding=(0, 0)
    ):
        super(GluConvTranspose2d, self).__init__()
        self.conv1 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            output_padding=output_padding,
        )
        self.conv2 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            output_padding=output_padding,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.sigmoid(self.conv2(x))
        out = out1 * out2
        return out


class EncoderDecoder(nn.Module):

    def __init__(self, temp_scale=10.0):
        super(EncoderDecoder, self).__init__()
        
        self.temp_scale = temp_scale

        self.input_projection = nn.Linear(in_features=257, out_features=256)

        self.conv1_enc = GluConv1d(in_channels=256, out_channels=128, 
                                    kernel_size=7, stride=1)
        self.conv2_enc = GluConv1d(in_channels=128, out_channels=256, 
                                    kernel_size=5, stride=1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8,
                                                dim_feedforward=256)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.encoder_linear = nn.Linear(in_features=256, out_features=2)

        decoder_layer = nn.TransformerDecoderLayer(d_model=256, nhead=8,
                                                dim_feedforward=256)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        
        self.conv1_dec = GluConv1d(in_channels=256, out_channels=128, 
                                    kernel_size=5, stride=1)
        self.conv2_dec = GluConv1d(in_channels=128, out_channels=256, 
                                    kernel_size=7, stride=1)
        
        self.decoder_linear = nn.Linear(in_features=256, out_features=257)

        self.bn1_enc = nn.BatchNorm1d(128)
        self.bn2_enc = nn.BatchNorm1d(256)
        self.bn3_enc = nn.BatchNorm1d(256)

        self.bn1_dec = nn.BatchNorm1d(128)
        self.bn2_dec = nn.BatchNorm1d(256)
        self.bn3_dec = nn.BatchNorm1d(256)

        self.sigmoid_activation = nn.Sigmoid()
        self.elu = nn.ELU(inplace=True)
        # self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=-1)
    
    
    def forward(self, x):

        # out = x
        projected_x = self.input_projection(x.permute(0,2,1))
        projected_x = projected_x.permute(0,2,1)
        # print("0. proj_x shape: ", projected_x.shape)

        e1_enc = self.elu(self.bn1_enc(self.conv1_enc(projected_x)))
        e2_enc = self.elu(self.bn2_enc(self.conv2_enc(e1_enc)))
        # print("1. e2_enc shape: ", e2_enc.shape)
        
        e2_enc = e2_enc.permute(2,0,1)
        e3_enc = self.transformer_encoder(e2_enc)
        # print("2. e3_enc shape: ", e3_enc.shape)
        
        e3_enc = e3_enc.permute(1,2,0)
        e3_enc = self.bn3_enc(e3_enc)
        posterior = self.encoder_linear(e3_enc.permute(0,2,1))
        # print("Posterior: ", posterior)
        
        posterior = self.softmax(posterior/self.temp_scale)
        sampled_val = gumbel_softmax(torch.log(posterior), 0.8)
        mask = sampled_val[:,:,1:2].repeat(1,1,256)
        # print("4. mask shape: ", mask.shape)

        enc_out = projected_x * mask.permute(0,2,1)
        enc_out = enc_out.permute(2,0,1)
        # print("5. enc_out shape: ", enc_out.shape)
        
        e1_dec = self.transformer_decoder(projected_x.permute(2,0,1), enc_out)
        e1_dec = e1_dec.permute(1,2,0)
        # print("6. e1_dec shape: ", e1_dec.shape)

        e2_dec = self.elu(self.bn1_dec(self.conv1_dec(e1_dec)))
        e3_dec = self.elu(self.bn2_dec(self.conv2_dec(e2_dec)))
        # print("7. e3_dec shape: ", e3_dec.shape)        

        e3_dec = self.bn3_dec(e3_dec)
        out = self.decoder_linear(e3_dec.permute(0,2,1))
        # print("8. out shape: ", out.shape)

        return posterior, sampled_val, out.permute(0,2,1)



if __name__ == "__main__":

    model = EncoderDecoder()
    model = model.cuda()
    x = torch.rand(4, 257, 300).to("cuda")
    p, s, o = model(x)
    print("posterior shape: ", p.shape)
    print("sample shape: ", s.shape)
    print("output shape: ", o.shape)























