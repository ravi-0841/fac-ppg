#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 17:10:07 2022

@author: ravi
"""

import os
import time
import math
import torch
import torch.nn as nn
import soundfile as sf
import numpy as np
import pylab

from torch.utils.data import DataLoader
from gcrn_unet import GCRN
from src.common.loss_function import SpectrogramL1Loss
from src.common.logger_unet import UnetLogger
from src.common.hparams_onflymixer import create_hparams
from src.common.hparams_onflyaligner import create_hparams as create_hparams_align
from pprint import pprint
from on_the_fly_mixer import OnTheFlyMixer
from on_the_fly_aligner import OnTheFlyAligner
from unet_trainer import load_checkpoint
from pesq import pesq
from pystoi import stoi

#%%
hparams = create_hparams()
hparams.validation_files = "/home/ravi/Desktop/fac-via-ppg/speechbrain_data/VESUS.txt"
# hparams.validation_files = "/home/ravi/Desktop/fac-via-ppg/speechbrain_data/LibriSpeech/test.txt"
hparams.noise_files = "/home/ravi/Desktop/fac-via-ppg/speechbrain_data/noise.txt"

dataloader = OnTheFlyMixer(
                            utterance_paths_file=hparams.validation_files,
                            noise_paths_file=hparams.noise_files,
                            hparams=hparams,
                            cutoff_length=3,
                            )


hparams_align = create_hparams_align()
hparams_align.validation_files = "./speechbrain_data/neutral_to_angry_valid.txt"
dataloader = OnTheFlyAligner(utterance_paths_file=hparams_align.validation_files,
                            hparams=hparams_align,
                            cutoff_length=3,
                            )

#%%
model = GCRN().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=hparams.weight_decay)
model, optimizer, _learning_rate, iteration = load_checkpoint("/home/ravi/Desktop/fac-via-ppg/unet_output_mixed_error_scratch_masked/checkpoint_237000", model, optimizer)

#%% For Enhancement
# q = np.random.choice(np.arange(0, len(dataloader)))
# data = dataloader[q]

# audio_src = data[2]
# audio_tar = data[3]
# src = data[0].to("cuda")
# src = src.unsqueeze(dim=0)
# src = src.permute(0,3,2,1)
# tar = data[1].to("cuda")
# tar = tar.unsqueeze(dim=0)
# tar = tar.permute(0,3,2,1)
# conv, embed = model(src)
# embed = embed.cpu().detach().numpy()
# conv = conv.permute(0,3,2,1)
# conv = conv.squeeze()
# audio_conv = torch.istft(conv, n_fft=hparams.n_fft, hop_length=hparams.hop_length, 
#                           win_length=hparams.win_length)
# sf.write("/home/ravi/Desktop/VESUS_src{}.wav".format(q), audio_src, 16000)
# sf.write("/home/ravi/Desktop/VESUS_tar{}.wav".format(q), audio_tar, 16000)
# audio_conv = audio_conv.cpu().detach().numpy()
# sf.write("/home/ravi/Desktop/VESUS_conv{}.wav".format(q), audio_conv, 16000)

# print("Noisy PESQ: ", pesq(16000, audio_tar, audio_src))
# print("Enhan PESQ: ", pesq(16000, audio_tar, audio_conv))

# print("Noisy STOI: ", stoi(audio_tar, audio_src, 16000))
# print("Enhan STOI: ", stoi(audio_tar, audio_conv, 16000))


#%% For Conversion
q = np.random.choice(np.arange(0, len(dataloader)))
data = dataloader[q]

stft_src = data[3].to("cuda")
stft_src = stft_src.unsqueeze(dim=0)
stft_src = stft_src.permute(0,3,2,1)
stft_tar = data[4].to("cuda")
stft_tar = stft_tar.unsqueeze(dim=0)
# stft_tar = stft_tar.permute(0,3,2,1)
stft_tar = stft_tar.squeeze()
stft_conv = model(stft_src)
stft_conv = stft_conv.permute(0,3,2,1)
stft_conv = stft_conv.squeeze()

stft_src = stft_src.permute(0,3,2,1).squeeze()
audio_conv = torch.istft(stft_conv, n_fft=hparams_align.n_fft, hop_length=hparams_align.hop_length, 
                          win_length=hparams_align.win_length)
audio_conv = audio_conv.cpu().detach().numpy()
sf.write("/home/ravi/Desktop/VESUS_conv_{}.wav".format(q), audio_conv, 16000)

audio_src = torch.istft(stft_src, n_fft=hparams_align.n_fft, hop_length=hparams_align.hop_length, 
                          win_length=hparams_align.win_length)
audio_src = audio_src.cpu().detach().numpy()
sf.write("/home/ravi/Desktop/VESUS_src_{}.wav".format(q), audio_src, 16000)

stft_conv = stft_conv[:,:,0]**2 + stft_conv[:,:,1]**2
stft_conv = stft_conv.cpu().detach().numpy()
pylab.subplot(121), pylab.imshow(np.log10(stft_conv), origin="lower")

stft_tar = stft_tar[:,:,0]**2 + stft_tar[:,:,1]**2
stft_tar = stft_tar.cpu().detach().numpy()
pylab.subplot(122), pylab.imshow(np.log10(stft_tar), origin="lower")























