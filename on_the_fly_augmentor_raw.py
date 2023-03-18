#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  28 15:25:43 2023

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
from torch.utils.data import DataLoader

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
                       lower_sr_factor=0.8, #0.8
                       upper_sr_factor=1.25): #1.25
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
        return clean_data.reshape(1,-1), random_sr
    
    
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
                torch.from_numpy(speech_data).float(), 
                torch.from_numpy(rating).float(),
                sr,
        )


    def __len__(self):
        return len(self.utterance_rating_paths)


def acoustics_collate_raw(batch):
    """Zero-pad the acoustic sequences in a mini-batch.

    Args:
        batch: An array with B elements, each is a tuple 
        (stft, speech_wav, tar_rating, sampling_rate).

    Returns:
        speech_padded: A (batch_size, 1, num_frames)
        tensor.
        lengths: A batch_size array, each containing the actual length
        of the input sequence.
    """
    # Right zero-pad all PPG sequences to max input length.
    # x is (PPG, acoustic), x[0] is PPG, which is an (L(varied), D) tensor.
    input_lengths, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([x[1].shape[1] for x in batch]), dim=0,
        descending=True)
    max_input_len = input_lengths[0]

    speech_padded = torch.FloatTensor(len(batch), 1, max_input_len)
    tar_ratings = torch.FloatTensor(len(batch), 5)
    
    speech_padded.zero_()
    tar_ratings.zero_()
    
    for i in range(len(ids_sorted_decreasing)):
        curr_speech = batch[ids_sorted_decreasing[i]][1]
        curr_tar_rate = batch[ids_sorted_decreasing[i]][2]

        speech_padded[i, :, :curr_speech.shape[1]] = curr_speech
        speech_padded[i, :, curr_speech.shape[1]:] = 0

        tar_ratings[i, :] = curr_tar_rate

    return speech_padded, tar_ratings, input_lengths


if __name__ == "__main__":
    hparams = create_hparams()

    dataclass = OnTheFlyAugmentor(
                                utterance_paths_file="./temp/train_fold_3.txt",
                                hparams=hparams,
                                augment=False,
                                )

    print(dataclass[10][1].shape)
    dataloader = DataLoader(
                            dataclass,
                            num_workers=1,
                            shuffle=False,
                            sampler=None,
                            batch_size=hparams.batch_size,
                            drop_last=True,
                            collate_fn=acoustics_collate_raw,
                            )
    for i, batch in enumerate(dataloader):
        print(batch[2].shape, torch.div(batch[2], 160, rounding_mode="floor"))
        print(batch[2].dtype, torch.div(batch[2], 160, rounding_mode="floor").dtype)
        break

    # src_stft, l, src = dataloader[19]

    # print("STFT shape: ", src_stft.shape)
    # print("length_array: ", l)
    # print("audio shape: ", src.shape)
    
    # pylab.figure()
    # pylab.imshow(np.log10(src_stft[:,:,0]**2 + src_stft[:,:,1]**2), origin="lower")
    # pylab.figure()
    # pylab.plot(src)





















