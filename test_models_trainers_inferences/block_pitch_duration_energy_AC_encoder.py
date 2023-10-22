#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 13:01:30 2023

@author: ravi
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable
from medianPool import MedianPool1d
from src.common.interpolation_block import (WSOLAInterpolation,
                                            BatchWSOLAInterpolation,
                                            BatchWSOLAInterpolationEnergy)
from on_the_fly_augmentor_raw_voice_mask import OnTheFlyAugmentor, acoustics_collate_raw
from src.common.loss_function import EntropyLoss
from src.common.hparams_onflyenergy_vesus import create_hparams
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


class ConvolutionalEncoderSmall(nn.Module):
    
    def __init__(self):
        super(ConvolutionalEncoderSmall, self).__init__()
        
        self.conv1_enc = nn.Conv1d(in_channels=1, out_channels=256, 
                                    kernel_size=10, stride=5, padding=2)
        self.conv2_enc = nn.Conv1d(in_channels=256, out_channels=512, 
                                    kernel_size=3, stride=2, padding=1)
        self.conv3_enc = nn.Conv1d(in_channels=512, out_channels=512, 
                                    kernel_size=3, stride=2, padding=1)
        self.conv4_enc = nn.Conv1d(in_channels=512, out_channels=256, 
                                    kernel_size=3, stride=2, padding=1)
        self.conv5_enc = nn.Conv1d(in_channels=256, out_channels=256, 
                                    kernel_size=3, stride=2, padding=1)
        self.conv6_enc = nn.Conv1d(in_channels=256, out_channels=256, 
                                    kernel_size=3, stride=2, padding=1)
        
        self.bn1_enc = nn.BatchNorm1d(256)
        self.bn2_enc = nn.BatchNorm1d(512)
        self.bn3_enc = nn.BatchNorm1d(512)
        self.bn4_enc = nn.BatchNorm1d(256)
        self.bn5_enc = nn.BatchNorm1d(256)
        self.bn6_enc = nn.BatchNorm1d(256)

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
        self.recurrent_layer = nn.LSTM(input_size=512, hidden_size=256, 
                                       num_layers=1, bidirectional=True, 
                                       dropout=0.2)
        self.bn = nn.BatchNorm1d(512)
        self.linear_layer = nn.Linear(in_features=512, out_features=2)
        self.median_pool = MedianPool1d(kernel_size=5, same=True)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, e):
        # x -> [batch, 512, #Time]
        # e - [batch, 1, #time//160]

        x, _ = self.recurrent_layer(x.permute(2,0,1))
        x = self.bn(x.permute(1,2,0))
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


class RatePredictorAC(nn.Module):
    def __init__(self, temp_scale=1.0): #50
        super(RatePredictorAC, self).__init__()
        self.thresh = nn.Threshold(-1e-7, -1)
        self.temp_scale = temp_scale

        self.conv_encoder = ConvolutionalEncoderSmall()
        
        self.value_linear1 = nn.Linear(in_features=256, out_features=128)
        self.value_linear2 = nn.Linear(in_features=128, out_features=1)

        self.emo_projection = nn.Linear(in_features=5, out_features=256)
        self.joint_projection = nn.Linear(in_features=256, out_features=256)
        self.bn_proj = nn.InstanceNorm1d(256)

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=256, 
                                                               nhead=4, 
                                                               dim_feedforward=256,
                                                               dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, 
                                                         num_layers=2)
        # self.bn_trans = nn.InstanceNorm1d(256)
        # self.bn_trans = nn.BatchNorm1d(256)
        
        self.linear_layer_rate = nn.Linear(in_features=256, out_features=11) #6
        self.linear_layer_pitch = nn.Linear(in_features=256, out_features=11) #6
        self.linear_layer_energy = nn.Linear(in_features=256, out_features=11) #6
        self.softmax = nn.Softmax(dim=-1)
        self.elu = nn.ELU(inplace=True)
    
    def forward(self, x, m, e): #(x, p, e)
        # x -> [batch, 1, #time]
        # e -> [batch, 5] one-hot encoding for [Neutral, Angry, Happy, Sad, Fearful]
        # m -> [batch, #time, 512] -> [batch, 512, #time]
        x = self.conv_encoder(x)

        m = -1*self.thresh(-1*m.permute(0,2,1))
        # m = m.permute(0,2,1)
        x = x + 10*x*m[:,:256,:]

        value = self.value_linear1(torch.max(x, dim=-1, keepdim=False)[0])
        value = self.elu(value)
        value = self.value_linear2(value)

        e_proj = self.emo_projection(e).unsqueeze(dim=-1)
        joint_x = x + e_proj #torch.cat((x, e_proj), dim=1)
        
        # joint_x -> [batch, 256, #time] -> [batch, #time, 256] -> [batch, 256, #time]
        x_proj = self.joint_projection(joint_x.permute(0,2,1)).permute(0,2,1)
        x_proj = x + self.elu(self.bn_proj(x_proj))
        
        # x_proj -> [batch, 256, #time] -> [#time, batch, 256]
        trans_out = self.transformer_encoder(x_proj.permute(2,0,1))
        trans_out = trans_out.permute(1,2,0)
        trans_out += e_proj/256.

        trans_out = torch.max(trans_out, dim=-1, keepdim=False)[0]
        output_rate = self.softmax(self.linear_layer_rate(trans_out)/self.temp_scale)
        output_pitch = self.softmax(self.linear_layer_pitch(trans_out)/self.temp_scale)
        output_energy = self.softmax(self.linear_layer_energy(trans_out)/self.temp_scale)
        
        return value, output_rate, output_pitch, output_energy


class MaskedRateModifier(nn.Module):
    def __init__(self, temp_scale=10.0):
        super(MaskedRateModifier, self).__init__()
        
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


# if __name__ == "__main__":

#     model_saliency = MaskedRateModifier()
#     model_rate = RatePredictor()
    
#     batch_size = 8
    
#     model_saliency = model_saliency.cuda()
#     model_rate = model_rate.cuda()
    
#     # Checking gradient operation
#     a_before = model_rate.linear_layer.weight.detach().cpu().numpy().copy()

#     num_params = sum(p.numel() for p in model_saliency.parameters() if p.requires_grad)
#     num_params += sum(p.numel() for p in model_rate.parameters() if p.requires_grad)
#     print("Total number of trainable parameters are: ", num_params)
    
#     # parameter optimization
#     optim1 = torch.optim.Adam(model_saliency.parameters(), lr=1e-5)
#     optim2 = torch.optim.Adam([{"params": model_saliency.conv_encoder.parameters()},
#                                {"params": model_rate.parameters()}], lr=1e-3)

#     # Criterion definition
#     criterion = nn.L1Loss()
#     criterion_ent = EntropyLoss()
    
#     # Set models in training mode
#     model_saliency.train()
#     model_rate.train()
    
#     WSOLA = BatchWSOLAInterpolationEnergy()
    
#     for epoch in range(1):
        
#         try:
#             # Input speech signal, ground truth saliency
#             input_speech = torch.rand(batch_size, 1, 16001).to("cuda")
#             e = torch.zeros((batch_size, 1, 16001)).to("cuda")
#             e[:,:,1300:-2000] = 1.0
#             target_saliency = torch.Tensor([[0.1, 0.3, 0.5, 0.0, 0.1]]).to("cuda")
#             target_saliency = target_saliency.repeat(batch_size, 1)
#             emotion_cats = torch.multinomial(torch.Tensor([0.2,0.2,0.2,0.2,0.2]), 
#                                              batch_size,
#                                              replacement=True)
#             emotion_codes = nn.functional.one_hot(emotion_cats, 5).float().to("cuda")
            
#             # Reset gradient tape
#             model_saliency.zero_grad()
#             model_rate.zero_grad()
             
#             f, p, m, s = model_saliency(input_speech, e)
            
#             # Compute Rate of modification
#             # r = model_rate(f.detach(), p.detach(), emotion_codes)
#             r = model_rate(f.detach(), m.detach(), emotion_codes)
    
#             # Printing shapes
#             # print("features shape: ", f.shape)
#             # print("posterior shape: ", p.shape)
#             # print("mask shape: ", m.shape)
#             # print("salience shape: ", s.shape)
#             # print("rate shape: ", r.shape)
        
#             # Interpolation check
#             # index = torch.multinomial(r, 1)
#             index = torch.argmax(r, dim=-1)
#             rate = 0.5 + 0.2*index
#             mod_speech, mod_e, _ = WSOLA(mask=m[:,:,0:1],
#                                     rate=rate,
#                                     speech=input_speech,
#                                     )
        
#             mod_speech = mod_speech.to("cuda")
#             mod_e = mod_e.to("cuda")

#             with torch.no_grad():
#             # model_saliency.eval()
#                 _, _, _, pred_sal = model_saliency(mod_speech, mod_e)
#             # model_saliency.train()
            
#             # Optimizing the models
#             loss_saliency = criterion(s, target_saliency)

#             loss_rate = torch.sum(torch.abs(pred_sal - emotion_codes), dim=-1)
#             loss_rate_idiotic = torch.mean(loss_rate.detach() * r.gather(1,index.view(-1,1)))
#             loss_rate_idiotic += 0.1 * criterion_ent(r)
#             corresp_probs = r.gather(1,index.view(-1,1)).view(-1)
#             loss_rate = torch.mean(torch.mul(loss_rate.detach(), torch.log(corresp_probs)))
#             loss_rate += 0.1 * criterion_ent(r)
            
#             print("Idiotic Loss: {}".format(loss_rate_idiotic.item()))
#             print("Correct Loss: {}".format(loss_rate.item()))
            
#             total_loss = loss_saliency + loss_rate
#             total_loss.backward()

#             grad_norm_rate = torch.nn.utils.clip_grad_norm_(
#                                                         model_rate.parameters(),
#                                                         1.0,
#                                                     )
#             grad_norm_sale = torch.nn.utils.clip_grad_norm_(
#                                                         model_saliency.parameters(),
#                                                         1.0,
#                                                     )
#             print("Salience Predictor grad_norm is: ", grad_norm_sale.item())
#             print("Rate Predictor grad_norm is: ", grad_norm_rate.item())
            
#             optim1.step()
#             optim2.step()
        
#         except Exception as ex:
#             print(ex)

#     # Checking gradient operation
#     a_after = model_rate.linear_layer.weight.detach().cpu().numpy().copy()


















