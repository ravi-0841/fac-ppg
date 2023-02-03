#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 15:57:43 2023

@author: ravi
"""


import os
import numpy as np
import soundfile as sf
import torch
import pylab
import pandas as pd
import ast
import librosa
from collections import Counter
import numpy as np

from src.common.hparams_onflychopper import create_hparams
from src.common.utils import load_filepaths

class OnTheFlyAugmentor():

    def __init__(
                    self, 
                    utterance_paths_file,
                    hparams,
                    augment=True,
                    cutoff_length=3,
                    base_folder="./speechbrain_data"
                ):
        
        self.utterance_rating_paths = load_filepaths(utterance_paths_file)
        self.base_folder = base_folder
        self.augment = augment
        self.hparams = hparams
        self.max_wav_len = self.hparams.sampling_rate * cutoff_length

        
    def _extract_stft_feats(self, data):
        data = torch.from_numpy(data).float()
        stft_features = torch.stft(
                                    data,
                                    n_fft=self.hparams.n_fft,
                                    win_length=self.hparams.win_length,
                                    hop_length=self.hparams.hop_length,
                                    return_complex=False,
                                    )
        
        return stft_features

    
    def _get_random_sr(self, 
                       lower_sr_factor=0.85, 
                       upper_sr_factor=1.15):
        act_sr = self.hparams.sampling_rate
        return int(act_sr * (lower_sr_factor + (upper_sr_factor - lower_sr_factor)*np.random.rand()))


    def _get_signal(self, path):
        clean_data, sr = sf.read(os.path.join(self.base_folder, path))

        # if sr != self.hparams.sampling_rate:
        #     clean_data = librosa.resample(clean_data, sr, self.hparams.sampling_rate)
        
        if self.augment:
            random_sr = self._get_random_sr()
        else:
            random_sr = self.hparams.sampling_rate

        clean_data = librosa.resample(clean_data, sr, random_sr)
        return clean_data, random_sr
    
    
    def _rating_structure(self, rating):
        rate_dict = {0:0, 1:0, 2:0, 3:0, 4:0}
        for r in rating:
            rate_dict[r] += 1
        return np.asarray(list(rate_dict.values())).reshape(1,-1) / 10.


    def __getitem__(self, index):
        path, rating = self.utterance_rating_paths[index].split(" ,")
        rating = ast.literal_eval(rating)
        speech_data, sr = self._get_signal(path)
        speech_stft = self._extract_stft_feats(speech_data)
        speech_stft = torch.sqrt(speech_stft[:,:,0]**2 + speech_stft[:,:,1]**2)
        rating = self._rating_structure(rating)
        return (
                speech_stft,
                speech_stft.shape[1], 
                torch.from_numpy(rating).float(),
                sr,
        ) #torch.from_numpy(speech_data).float()


    def __len__(self):
        return len(self.utterance_rating_paths)


if __name__ == "__main__":
    hparams = create_hparams()

    dataloader = OnTheFlyAugmentor(
                                utterance_paths_file="/home/ravi/Desktop/VESUS_saliency.txt",
                                hparams=hparams,
                                )

    # src_stft, l, src = dataloader[19]

    # print("STFT shape: ", src_stft.shape)
    # print("length_array: ", l)
    # print("audio shape: ", src.shape)
    
    # pylab.figure()
    # pylab.imshow(np.log10(src_stft[:,:,0]**2 + src_stft[:,:,1]**2), origin="lower")
    # pylab.figure()
    # pylab.plot(src)