#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 12:05:27 2023

@author: ravi
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from medianPool import MedianPool1d
from on_the_fly_augmentor_raw_voice_mask import OnTheFlyAugmentor, acoustics_collate_raw
from src.common.loss_function import EntropyLoss
from src.common.hparams_onflyenergy import create_hparams
from src.common.utils import load_filepaths
from torch.utils.data import DataLoader


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


class ConvolutionalEncoder(nn.Module):
    
    def __init__(self):
        super(ConvolutionalEncoder, self).__init__()
        
        self.conv1_enc = nn.Conv1d(in_channels=1, out_channels=256, 
                                    kernel_size=10, stride=5, padding=2)
        self.conv2_enc = nn.Conv1d(in_channels=256, out_channels=256, 
                                    kernel_size=3, stride=2, padding=1)
        self.conv3_enc = nn.Conv1d(in_channels=256, out_channels=256, 
                                    kernel_size=3, stride=2, padding=1)
        self.conv4_enc = nn.Conv1d(in_channels=256, out_channels=512, 
                                    kernel_size=3, stride=2, padding=1)
        self.conv5_enc = nn.Conv1d(in_channels=512, out_channels=512, 
                                    kernel_size=3, stride=2, padding=1)
        self.conv6_enc = nn.Conv1d(in_channels=512, out_channels=512, 
                                    kernel_size=3, stride=2, padding=1)
        
        self.bn1_enc = nn.BatchNorm1d(256)
        self.bn2_enc = nn.BatchNorm1d(256)
        self.bn3_enc = nn.BatchNorm1d(256)
        self.bn4_enc = nn.BatchNorm1d(512)
        self.bn5_enc = nn.BatchNorm1d(512)
        self.bn6_enc = nn.BatchNorm1d(512)

        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        # x -> [#batch, 1, #Time]
        e1_enc = self.elu(self.bn1_enc(self.conv1_enc(x)))
        e2_enc = self.elu(self.bn2_enc(self.conv2_enc(e1_enc)))
        e3_enc = self.elu(self.bn3_enc(self.conv3_enc(e2_enc)))
        e4_enc = self.elu(self.bn4_enc(self.conv4_enc(e3_enc)))
        e5_enc = self.elu(self.bn5_enc(self.conv5_enc(e4_enc)))
        e6_enc = self.elu(self.bn6_enc(self.conv6_enc(e5_enc)))
        
        return e6_enc


# class SaliencePredictor(nn.Module):
#     def __init__(self):
#         super(SaliencePredictor, self).__init__()

#         transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=512, 
#                                                                 nhead=8, 
#                                                                 dim_feedforward=512)
#         self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, 
#                                                           num_layers=3)
        
#         self.recurrent_layer = nn.LSTM(input_size=512, hidden_size=256, 
#                                         num_layers=2, bidirectional=True, 
#                                         dropout=0.2)
#         self.bn_transformer = nn.BatchNorm1d(512)
#         self.bn_recurrent = nn.BatchNorm1d(512)
#         self.linear_layer = nn.Linear(in_features=512, out_features=5)
#         self.softmax = nn.Softmax(dim=-1)
    
#     def forward(self, x, m):
#         # x -> [batch, 512, #time]
#         # m -> [batch, 512, #time]
#         t1_enc = self.transformer_encoder(x.permute(2,0,1))
#         # t1_enc -> [#time, batch, 512] -> [batch, 512, #time]
#         t1_enc = t1_enc.permute(1,2,0)
#         t1_enc = self.bn_transformer(t1_enc)
#         t1_enc = t1_enc * m

#         lstm_out, _ = self.recurrent_layer(t1_enc.permute(2,0,1))
#         lstm_out = lstm_out[-1, :, :]
#         lstm_out = self.bn_recurrent(lstm_out)
#         output = self.softmax(self.linear_layer(lstm_out))
#         return output


class SaliencePredictor(nn.Module):
    def __init__(self):
        super(SaliencePredictor, self).__init__()

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=512, 
                                                                nhead=8, 
                                                                dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, 
                                                          num_layers=3)
        self.bn_transformer = nn.BatchNorm1d(512)
        self.linear_layer = nn.Linear(in_features=512, out_features=5)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, m):
        # x -> [batch, 512, #time]
        # m -> [batch, 512, #time]
        t1_enc = self.transformer_encoder(x.permute(2,0,1))
        # t1_enc -> [#time, batch, 512] -> [batch, 512, #time]
        t1_enc = t1_enc.permute(1,2,0)
        t1_enc = self.bn_transformer(t1_enc)
        t1_enc = t1_enc * m
        
        t1_enc = torch.mean(t1_enc, dim=-1)

        output = self.softmax(self.linear_layer(t1_enc))
        return output


class MaskGenerator(nn.Module):
    def __init__(self, temp_scale=10.0):
        super(MaskGenerator, self).__init__()
        self.temp_scale = temp_scale
        # self.recurrent_layer = nn.LSTM(input_size=512, hidden_size=512, 
        #                                num_layers=1, bidirectional=False, 
        #                                dropout=0)
        # self.bn = nn.BatchNorm1d(512)
        self.linear_layer = nn.Linear(in_features=512, out_features=2)
        self.median_pool = MedianPool1d(kernel_size=5, same=True)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, e):
        # x -> [batch, 512, #Time]
        # e - [batch, 1, #time//160]

        # x, _ = self.recurrent_layer(x.permute(2,0,1))
        # x = self.bn(x.permute(1,2,0))
        posterior = self.linear_layer(x.permute(0,2,1))        
        posterior = self.softmax(posterior/self.temp_scale)

        # if not self.training:
        #     sampled_val = torch.bernoulli(posterior)
        # else:
        sampled_val = gumbel_softmax(torch.log(posterior), 0.8)

        sampled_val = torch.mul(sampled_val[:,:,1:2].permute(0,2,1), 
                                e[:,:,:sampled_val.size(1)])

        mask = self.median_pool(sampled_val)
        mask = self.median_pool(self.median_pool(mask))
        return posterior, mask


class MaskedSaliencePredictor(nn.Module):
    def __init__(self, temp_scale=10.0):
        super(MaskedSaliencePredictor, self).__init__()
        
        self.temp_scale = temp_scale

        self.conv_encoder = ConvolutionalEncoder()
        
        self.mask_generator = MaskGenerator(temp_scale=self.temp_scale)
        
        self.salience_predictor = SaliencePredictor()
        # self.rate_predictor = RatePredictor()

        self.sigmoid_activation = nn.Sigmoid()
        self.elu = nn.ELU(inplace=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, e, pre_computed_mask=None, use_posterior=False):
        # x shape - [batch, 1, #time]
        # e - [batch, 1, #time//160]
        # pre_computed_mask shape - [batch, #time, 512]

        # conv_features -> [batch, 512, #time]
        conv_features = self.conv_encoder(x)

        posterior, mask = self.mask_generator(conv_features, e)
        # mask = torch.mul(mask, e[:,:,:mask.size(2)])
        mask = mask.repeat(1,512,1)

        if pre_computed_mask is not None:
            mask = pre_computed_mask

        if not use_posterior:
            salience = self.salience_predictor(conv_features, mask)
        else:
            posterior_mask = posterior[:,:,1:2].permute(0,2,1).repeat(1,512,1)
            salience = self.salience_predictor(conv_features, posterior_mask)

        return conv_features, posterior, mask.permute(0,2,1), salience


if __name__ == "__main__":

    model_saliency = MaskedSaliencePredictor()    
    model_saliency = model_saliency.cuda()
    num_params = sum(p.numel() for p in model_saliency.parameters() if p.requires_grad)
    print("Total number of trainable parameters are: ", num_params)
    
    # Dataloader
    hparams = create_hparams()
    dataclass = OnTheFlyAugmentor(
                                utterance_paths_file="./speechbrain_data/VESUS_saliency_training_small.txt",
                                hparams=hparams,
                                augment=True,
                                )
    dataloader = DataLoader(
                            dataclass,
                            num_workers=1,
                            shuffle=True,
                            sampler=None,
                            batch_size=8,
                            drop_last=True,
                            collate_fn=acoustics_collate_raw,
                            )
    
    # parameter optimization
    optimizer = torch.optim.Adam(model_saliency.parameters(), lr=1e-5)

    # Criterion definition
    criterion = nn.L1Loss()
    
    # Set models in training mode
    model_saliency.train()
    
    for i, batch in enumerate(dataloader):
        
        # Reset gradient tape
        model_saliency.zero_grad()
        
        (x, e, y, l) = (batch[0].to("cuda"), batch[1].to("cuda"),
                      batch[2].to("cuda"), batch[3])

        # Input speech signal, ground truth saliency
        _, _, masks, y_hat = model_saliency(x, e)
        
        # Optimizing the models
        loss_saliency = criterion(y_hat, y)
        loss_saliency.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
                                                    model_saliency.parameters(),
                                                    1.0,
                                                )
        print("Loss: {}, grad_norm: {}".format(loss_saliency.item(), grad_norm.item()))
        optimizer.step()
        
        # print(masks.shape, e.shape)
        
        if i > 10:
            break


















