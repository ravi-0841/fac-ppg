#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 17:20:41 2023

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


class SaliencyPredictor(nn.Module):

    def __init__(self, temp_scale=10.0):
        super(SaliencyPredictor, self).__init__()
        
        self.temp_scale = temp_scale

        self.input_projection = nn.Linear(in_features=257, out_features=512)

        self.conv1_enc = GluConv1d(in_channels=512, out_channels=1024, 
                                    kernel_size=7, stride=1)
        self.conv2_enc = GluConv1d(in_channels=1024, out_channels=512, 
                                    kernel_size=5, stride=1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8,
                                                dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.encoder_linear = nn.Linear(in_features=512, out_features=2)
        
        self.recurrent_layer = nn.LSTM(input_size=512, hidden_size=256, 
                               num_layers=1, bidirectional=True, dropout=0.2)

        self.decoder_linear = nn.Linear(in_features=512, out_features=5)

        self.bn1_enc = nn.BatchNorm1d(1024)
        self.bn2_enc = nn.BatchNorm1d(512)
        self.bn3_enc = nn.BatchNorm1d(512)

        self.bn1_dec = nn.BatchNorm1d(512)

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
        mask = sampled_val[:,:,1:2].repeat(1,1,512) # 256 for smaller model
        # print("4. mask shape: ", mask.shape)

        enc_out = projected_x * mask.permute(0,2,1)
        # enc_out = e3_enc * mask.permute(0,2,1)
        enc_out = enc_out.permute(2,0,1)
        # print("5. enc_out shape: ", enc_out.shape)
        
        lstm_out, _ = self.recurrent_layer(enc_out)
        lstm_out = lstm_out[-1, :, :]
        print("6. lstm_out shape: ", lstm_out.shape)

        lstm_out = self.bn1_dec(lstm_out)
        out = self.softmax(self.decoder_linear(lstm_out))
        print("8. out shape: ", out.shape)

        return posterior, sampled_val, out


if __name__ == "__main__":

    model = SaliencyPredictor()
    model = model.cuda()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of trainable parameters are: ", num_params)
    x = torch.rand(4, 257, 300).to("cuda")
    p, s, o = model(x)
    print("posterior shape: ", p.shape)
    print("sample shape: ", s.shape)
    print("output shape: ", o.shape)




















