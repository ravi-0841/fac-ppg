#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:35:28 2023

@author: ravi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from medianPool import MedianPool1d
from interpolation_block import WSOLAInterpolation


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


class ConvolutionalTransformerEncoder(nn.Module):
    
    def __init__(self):
        super(ConvolutionalTransformerEncoder, self).__init__()
        
        self.conv1_enc = nn.Conv1d(in_channels=1, out_channels=512, 
                                    kernel_size=10, stride=5, padding=2)
        self.conv2_enc = nn.Conv1d(in_channels=512, out_channels=512, 
                                    kernel_size=3, stride=2, padding=1)
        self.conv3_enc = nn.Conv1d(in_channels=512, out_channels=512, 
                                    kernel_size=3, stride=2, padding=1)
        self.conv4_enc = nn.Conv1d(in_channels=512, out_channels=512, 
                                    kernel_size=3, stride=2, padding=1)
        self.conv5_enc = nn.Conv1d(in_channels=512, out_channels=512, 
                                    kernel_size=3, stride=2, padding=1)
        self.conv6_enc = nn.Conv1d(in_channels=512, out_channels=512, 
                                    kernel_size=3, stride=2, padding=1)

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=512, 
                                                               nhead=8, 
                                                               dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, 
                                                         num_layers=2)
        
        self.bn1_enc = nn.BatchNorm1d(512)
        self.bn2_enc = nn.BatchNorm1d(512)
        self.bn3_enc = nn.BatchNorm1d(512)
        self.bn4_enc = nn.BatchNorm1d(512)
        self.bn5_enc = nn.BatchNorm1d(512)
        self.bn6_enc = nn.BatchNorm1d(512)
        
        self.bn_transformer = nn.BatchNorm1d(512)

        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        # x -> [#batch, 1, #Time]
        e1_enc = self.elu(self.bn1_enc(self.conv1_enc(x)))
        e2_enc = self.elu(self.bn2_enc(self.conv2_enc(e1_enc)))
        e3_enc = self.elu(self.bn3_enc(self.conv3_enc(e2_enc)))
        e4_enc = self.elu(self.bn4_enc(self.conv4_enc(e3_enc)))
        e5_enc = self.elu(self.bn5_enc(self.conv5_enc(e4_enc)))
        e6_enc = self.elu(self.bn6_enc(self.conv6_enc(e5_enc)))
        
        conv_features = e6_enc.permute(2,0,1)
        t1_enc = self.transformer_encoder(conv_features)
        
        t1_enc = t1_enc.permute(1,2,0)
        t1_enc = self.bn_transformer(t1_enc)
        return t1_enc


class SaliencePredictor(nn.Module):
    def __init__(self):
        super(SaliencePredictor, self).__init__()
        self.recurrent_layer = nn.LSTM(input_size=512, hidden_size=256, 
                                       num_layers=2, bidirectional=True, 
                                       dropout=0.2)
        # self.bn = nn.BatchNorm1d(512)
        self.linear_layer = nn.Linear(in_features=512, out_features=5)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        # x -> [#time, batch, #dimension]
        lstm_out, _ = self.recurrent_layer(x)
        lstm_out = lstm_out[-1, :, :]
        # lstm_out = self.bn(lstm_out)
        output = self.softmax(self.linear_layer(lstm_out))
        return output


class RatePredictor(nn.Module):
    def __init__(self):
        super(RatePredictor, self).__init__()
        self.recurrent_layer = nn.LSTM(input_size=512, hidden_size=256, 
                                       num_layers=2, bidirectional=True, 
                                       dropout=0.2)
        # self.bn = nn.BatchNorm1d(512)
        self.linear_layer = nn.Linear(in_features=512, out_features=7)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        # x -> [batch, #dimension, #time] -> [#time, #batch, #dimension]
        x = x.permute(2,0,1)
        lstm_out, _ = self.recurrent_layer(x)
        lstm_out = lstm_out[-1, :, :]
        # lstm_out = self.bn(lstm_out)
        output = self.softmax(self.linear_layer(lstm_out))
        return output


class MaskGenerator(nn.Module):
    def __init__(self, temp_scale=10.0):
        super(MaskGenerator, self).__init__()
        self.temp_scale = temp_scale
        self.linear_layer = nn.Linear(in_features=512, out_features=2)
        self.median_pool = MedianPool1d(kernel_size=5, same=True)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        # x -> [batch, 512, #Time]
        posterior = self.linear_layer(x.permute(0,2,1))        
        posterior = self.softmax(posterior/self.temp_scale)
        sampled_val = gumbel_softmax(torch.log(posterior), 0.8)
        mask = self.median_pool(sampled_val[:,:,1:2].permute(0,2,1))
        mask = self.median_pool(self.median_pool(mask))
        return posterior, mask


class MaskedRateModifier(nn.Module):
    def __init__(self, temp_scale=10.0):
        super(MaskedRateModifier, self).__init__()
        
        self.temp_scale = temp_scale

        self.conv_trans_encoder = ConvolutionalTransformerEncoder()
        
        self.mask_generator = MaskGenerator(temp_scale=self.temp_scale)
        
        self.salience_predictor = SaliencePredictor()
        # self.rate_predictor = RatePredictor()

        self.sigmoid_activation = nn.Sigmoid()
        self.elu = nn.ELU(inplace=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, pre_computed_mask=None):
        # x shape - [batch, 1, #time]
        # pre_computed_mask shape - [batch, #time, 512]

        # print("0. x shape: ", x.shape)

        conv_trans_features = self.conv_trans_encoder(x)
        # print("1. transformer_features shape: ", transformer_features.shape)

        posterior, mask = self.mask_generator(conv_trans_features)
        mask = mask.permute(0,2,1).repeat(1,1,512) #256 for small model
        # print("2. mask shape: ", mask.shape)

        if pre_computed_mask is not None:
            mask = pre_computed_mask

        enc_out = conv_trans_features * mask.permute(0,2,1)
        # print("3. enc_out shape: ", enc_out.shape)

        # rate = self.rate_predictor(conv_trans_features)
        salience = self.salience_predictor(enc_out.permute(2,0,1))

        return conv_trans_features, posterior, mask, salience


if __name__ == "__main__":

    model_saliency = MaskedRateModifier()
    model_rate = RatePredictor()
    
    model_saliency = model_saliency.cuda()
    model_rate = model_rate.cuda()
    
    # Checking gradient operation
    a_before = model_rate.linear_layer.weight.detach().cpu().numpy().copy()

    num_params = sum(p.numel() for p in model_saliency.parameters() if p.requires_grad)
    num_params += sum(p.numel() for p in model_rate.parameters() if p.requires_grad)
    print("Total number of trainable parameters are: ", num_params)
    
    # parameter optimization
    optim1 = torch.optim.Adam(model_saliency.parameters(), lr=1)
    optim2 = torch.optim.Adam(model_rate.parameters(), lr=1)

    # Criterion definition
    criterion = nn.L1Loss()
    
    for epoch in range(1):
        
        # Set models in training mode
        model_saliency.train()
        model_rate.train()
        
        # Input speech signal, ground truth saliency
        input_speech = torch.rand(1, 1, 16001).to("cuda")
        target_saliency = torch.Tensor([[0.1, 0.3, 0.5, 0.0, 0.1]]).to("cuda")
        
        # Sample intended saliency
        index_intent = torch.multinomial(torch.Tensor([0.2, 0.2, 0.2, 0.2, 0.2]), 1)
        intent_saliency = torch.zeros(1,5)
        intent_saliency[0, index_intent[0]] = 1.0
        intent_saliency = intent_saliency.to("cuda")
        # intent_saliency = torch.Tensor([[0.0, 1.0, 0.0, 0.0, 0.0]]).to("cuda")
        
        # Reset gradient tape
        model_saliency.zero_grad()
        model_rate.zero_grad()
         
        f, p, m, s = model_saliency(input_speech)
    
        # Optimizing the saliency_predictor model
        loss_salience = criterion(s, target_saliency)
        loss_salience.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
                                                    model_saliency.parameters(),
                                                    1.0,
                                                )
        print("grad_norm is: ", grad_norm)
        print(model_saliency.conv_trans_encoder.conv1_enc.weight.grad)
        optim1.step()
        
        # Compute Rate of modification
        r = model_rate(f.detach()) # use detach operation to prevent backprop through feature extractor

        # Printing shapes
        # print("features shape: ", f.shape)
        # print("posterior shape: ", p.shape)
        # print("mask shape: ", m.shape)
        # print("salience shape: ", s.shape)
        # print("rate shape: ", r.shape)
    
        # Interpolation check
        WSOLA = WSOLAInterpolation()
        index = torch.multinomial(r, 1)
        rate = 0.7 + 0.1*index[0][0]
        mod_speech, x, y = WSOLA(mask=m[:,:,0],
                                rate=rate,
                                speech=input_speech,
                                )
    
        mod_speech = mod_speech.to("cuda")
        # with torch.no_grad():
        model_saliency.eval()
        with torch.no_grad():
            _, _, _, s = model_saliency(mod_speech)
    
        loss = criterion(s, intent_saliency)
        loss_rate = loss.detach() * r[0,index[0][0]]
        loss_rate.backward()
        optim2.step()

    # Checking gradient operation
    a_after = model_rate.linear_layer.weight.detach().cpu().numpy().copy()



















