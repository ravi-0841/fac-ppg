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
                    base_folder=""
                ):
        
        self.utterance_rating_paths = load_filepaths(utterance_paths_file)
        self.base_folder = base_folder
        self.augment = augment
        self.hparams = hparams

        
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
                       lower_sr_factor=0.8, #0.85
                       upper_sr_factor=1.25): #1.15
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

        clean_data = librosa.resample(clean_data, orig_sr=sr, target_sr=random_sr)
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


def acoustics_collate(batch):
    """Zero-pad the acoustic sequences in a mini-batch.

    Args:
        batch: An array with B elements, each is a tuple 
        (stft, len(stft), tar_rating, sampling_rate).

    Returns:
        src_stft_padded: A (batch_size, feature_dim_1, num_frames_1)
        tensor.
        lengths: A batch_size array, each containing the actual length
        of the input sequence.
    """
    # Right zero-pad all PPG sequences to max input length.
    # x is (PPG, acoustic), x[0] is PPG, which is an (L(varied), D) tensor.
    input_lengths, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([x[0].shape[1] for x in batch]), dim=0,
        descending=True)
    max_input_len = input_lengths[0]
    stft_dim = batch[0][0].shape[0]

    src_stft_padded = torch.FloatTensor(len(batch), stft_dim, max_input_len)
    tar_ratings = torch.FloatTensor(len(batch), 5)
    
    src_stft_padded.zero_()
    tar_ratings.zero_()
    
    for i in range(len(ids_sorted_decreasing)):
        curr_src_stft = batch[ids_sorted_decreasing[i]][0]
        curr_tar_rate = batch[ids_sorted_decreasing[i]][2]

        src_stft_padded[i, :, :curr_src_stft.shape[1]] = curr_src_stft
        src_stft_padded[i, :, curr_src_stft.shape[1]:] = curr_src_stft[:, -2:-1]

        tar_ratings[i, :] = curr_tar_rate

    return src_stft_padded, tar_ratings, input_lengths


if __name__ == "__main__":
    hparams = create_hparams()

    dataloader = OnTheFlyAugmentor(
                                utterance_paths_file="./speechbrain_data/VESUS_saliency_training.txt",
                                hparams=hparams,
                                augment=False,
                                )

    # src_stft, l, src = dataloader[19]

    # print("STFT shape: ", src_stft.shape)
    # print("length_array: ", l)
    # print("audio shape: ", src.shape)
    
    # pylab.figure()
    # pylab.imshow(np.log10(src_stft[:,:,0]**2 + src_stft[:,:,1]**2), origin="lower")
    # pylab.figure()
    # pylab.plot(src)





















