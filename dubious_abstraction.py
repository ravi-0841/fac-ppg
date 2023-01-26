#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 18:12:46 2022

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
import librosa.display
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from wav2vec_knowledge_distillation import Wav2Vec2_encoder, Wav2Vec2_pretrained
from src.common.loss_function import SpectrogramL1Loss
from src.common.logger_wav2vec import Wav2VecLogger
from src.common.hparams_onflychopper import create_hparams
from pprint import pprint
from on_the_fly_chopper import OnTheFlyChopper
from wav2vec_distillation_trainer import load_checkpoint
from pesq import pesq
from pystoi import stoi

from stft import STFT
from typing import Dict, Any, Tuple
from GCRN_encoder_decoder import GCRN_embedding_decoder


#%%
class GCRNNet_pre_train_embedding_decoder(nn.Module):
    def __init__(self, net_args: Dict[str, Any], mixing_alpha=0) -> None:
        super(GCRNNet_pre_train_embedding_decoder, self).__init__()
        self.stft = STFT(
            win_size=net_args["win_size"],
            hop_size=net_args["hop_size"],
            fft_size=net_args["fft_size"],
            win_type=net_args["win_type"],
        )

        self.gcrn = GCRN_embedding_decoder(mixing_alpha=mixing_alpha)

    def forward(
        self, x: torch.Tensor, input_embed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_stft = self.stft(x, mode="stft")
        output_stft, embeds = self.gcrn(input_stft, input_embed)
        output_audio = self.stft(output_stft, mode="istft")
        return input_stft, output_stft, output_audio


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
pt_model = Wav2Vec2_pretrained(config_path=hparams.config_path)

model = Wav2Vec2_encoder(config_path=hparams.config_path)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=hparams.weight_decay)
model, optimizer, _learning_rate, iteration = load_checkpoint("/home/ravi/Desktop/fac-via-ppg/wav2vec_output/checkpoint_53200", model, optimizer)


#%%
net_args = {"win_size":400, "hop_size":320, "fft_size":512, "win_type":"hann"}
embed_decoder = GCRNNet_pre_train_embedding_decoder(
                                            net_args=net_args, 
                                            mixing_alpha=0,
                                        )
ckpt_dict = torch.load("/home/ravi/Desktop/Meta_Internship/models/clean_embedL1_frozen_false_latest_model.pt", map_location="cpu")
embed_decoder.load_state_dict(ckpt_dict)


#%%

# diff_embed = []
# for i in range(len(dataloader)):

q = np.random.choice(np.arange(0, len(dataloader)))
data = dataloader[q]

audio = data[1].reshape(1,-1)
audio = audio.to("cuda")

e_pt = pt_model(audio)
e_pt = e_pt.cpu().detach().numpy()

e_dt = model(audio)
e_dt = e_dt.cpu().detach().numpy()

# diff_embed.append(np.mean(np.abs(e_pt - e_dt)**2))


#%%
embed = pt_model(audio)
embed = embed.squeeze().cpu().detach().numpy()

# path = os.path.join("/home/ravi/Desktop/Meta_Internship/features_w2v2_distilled", 
#                     os.path.basename(dataloader.utterance_paths[q])[:-5]+"_conv_trans_feats.h5context")

# with h5py.File(path, "w") as f:
#     f["trans_layer_11"] = embed
#     f.close()

# pylab.figure()
# pylab.imshow(embed.T)

embed = torch.from_numpy(embed).unsqueeze(dim=0)
embed = embed.unsqueeze(dim=2)
embed_decoder = embed_decoder.to("cuda")
audio = audio.to("cuda")
embed = embed.to("cuda")

input_, output_, gen_audio = embed_decoder(audio, embed)
input_ = input_.cpu().detach().numpy().squeeze()
input_ = input_[0]**2 + input_[1]**2

pylab.figure()
pylab.imshow(np.log10(input_.T), origin="lower")
pylab.title("Ground truth STFT")
pylab.tick_params(size=10)
pylab.tick_params(width=2)
pylab.tick_params(axis='both', which='major', labelsize=13)

output_ = output_.cpu().detach().numpy().squeeze()
output_ = output_[0]**2 + output_[1]**2
pylab.figure(), pylab.imshow(np.log10(output_.T), origin="lower")
pylab.title("Pretrained generated STFT")
pylab.tick_params(size=10)
pylab.tick_params(width=2)
pylab.tick_params(axis='both', which='major', labelsize=13)















