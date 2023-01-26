#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 13:40:47 2022

@author: ravi
"""

import numpy as np
import soundfile as sf
import torch
import pylab

from src.common.hparams_onflymixer import create_hparams
from src.common.utils import load_filepaths

class OnTheFlyChopper():

    def __init__(
                    self, 
                    utterance_paths_file,
                    hparams,
                    cutoff_length=5,
                ):
        
        self.utterance_paths = load_filepaths(utterance_paths_file)
        self.hparams = hparams
        
        self.max_wav_len = self.hparams.sampling_rate * cutoff_length

        
    def extract_stft_feats(self, data):
        data = torch.from_numpy(data).float()
        stft_features = torch.stft(
                                    data,
                                    n_fft=self.hparams.n_fft,
                                    win_length=self.hparams.win_length,
                                    hop_length=self.hparams.hop_length,
                                    return_complex=False,
                                    )
        
        return stft_features

    
    def compute_scale(self, signal_energy, noise_energy, snr_db):
        return np.sqrt(signal_energy / noise_energy) * (10 ** (-1*snr_db / 20))


    def chop_signal(self, index):
        clean_file_path = self.utterance_paths[index]
        clean_data, sr = sf.read(clean_file_path)
        
        if len(clean_data) > self.max_wav_len:
            start = np.random.randint(0, len(clean_data) - self.max_wav_len)
            clean_data = clean_data[start:start+self.max_wav_len]
        else:
            diff = self.max_wav_len - len(clean_data)
            clean_data = np.concatenate(
                                        (clean_data, np.zeros((diff,))),
                                        axis=0,
                                        )
        return clean_data


    def __getitem__(self, index):
        speech_data = self.chop_signal(index)
        speech_stft = self.extract_stft_feats(speech_data)
        return speech_stft, torch.from_numpy(speech_data).float()


    def __len__(self):
        return len(self.utterance_paths)


if __name__ == "__main__":
    hparams = create_hparams()

    dataloader = OnTheFlyChopper(
                                utterance_paths_file=hparams.validation_files,
                                hparams=hparams,
                                )

    src_stft, src = dataloader[19]
    
    pylab.figure()
    pylab.imshow(np.log10(src_stft[:,:,0]**2 + src_stft[:,:,1]**2), origin="lower")
    pylab.figure()
    pylab.plot(src)




















        
