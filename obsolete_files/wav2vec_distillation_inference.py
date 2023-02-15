#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 12:34:33 2022

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
import h5py
from torch.utils.data import DataLoader
from wav2vec_knowledge_distillation import Wav2Vec2_encoder
from src.common.loss_function import SpectrogramL1Loss
from src.common.logger_wav2vec import Wav2VecLogger
from src.common.hparams_onflychopper import create_hparams
from pprint import pprint
from on_the_fly_chopper import OnTheFlyChopper
from wav2vec_distillation_trainer import load_checkpoint
from pesq import pesq
from pystoi import stoi

#%%
hparams = create_hparams()
# hparams.validation_files = "/home/ravi/Desktop/fac-via-ppg/speechbrain_data/VESUS.txt"
# hparams.validation_files = "/home/ravi/Desktop/fac-via-ppg/speechbrain_data/LibriSpeech/test.txt"
hparams.validation_files = "/home/ravi/Desktop/Meta_Internship/wav2vec_feats.txt"

dataloader = OnTheFlyChopper(utterance_paths_file=hparams.validation_files,
                            hparams=hparams,
                            cutoff_length=5,
                            )

#%%
model = Wav2Vec2_encoder(config_path=hparams.config_path)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=hparams.weight_decay)
model, optimizer, _learning_rate, iteration = load_checkpoint("/home/ravi/Desktop/fac-via-ppg/wav2vec_output/checkpoint_53200", model, optimizer)

#%%
for i in range(3):

    q = np.random.choice(np.arange(0, len(dataloader)))
    data = dataloader[q]
    
    audio = data[1].reshape(1,-1)
    audio = audio.to("cuda")
    embed = model(audio)
    embed = embed.squeeze().cpu().detach().numpy()
    
    path = os.path.join("/home/ravi/Desktop/Meta_Internship/features_w2v2_distilled", 
                        os.path.basename(dataloader.utterance_paths[q])[:-5]+"_conv_trans_feats.h5context")
    
    with h5py.File(path, "w") as f:
        f["trans_layer_11"] = embed
        f.close()
    
    pylab.figure()
    pylab.imshow(embed.T)
    
    
    # q = np.random.choice(np.arange(0, len(dataloader)))
    # data = dataloader[q]
    
    # audio = data[1].reshape(1,-1)
    # audio = audio.to("cuda")
    # embed2 = model(audio)
    # embed2 = embed2.squeeze().cpu().detach().numpy()
    
    # pylab.figure()
    # pylab.imshow(embed2.T)
    # break




















