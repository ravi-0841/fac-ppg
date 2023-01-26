#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 15:55:23 2022

@author: ravi
"""

import numpy as np
import soundfile as sf
import torch
import pylab

from src.common.hparams_onflymixer import create_hparams
from src.common.utils import load_filepaths

class OnTheFlyMixer():

    def __init__(
                    self, 
                    utterance_paths_file,
                    hparams,
                    noise_paths_file=None,
                    cutoff_length=5,
                ):
        
        self.utterance_paths = load_filepaths(utterance_paths_file)
        self.hparams = hparams
        
        if noise_paths_file is not None:
            self.noise_paths = load_filepaths(noise_paths_file)
        else:
            self.noise_paths = None
        
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


    def mix_noise(self, index):
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
        
        sample_snr_db = 5*np.random.rand()  # Range is 0-5db

        if self.noise_paths is not None:
            if np.random.rand() > 0.25:
                noise_file_path = np.random.choice(self.noise_paths)
                noise_data, sr = sf.read(noise_file_path)
                if len(noise_data) > self.max_wav_len:
                    noise_data = noise_data[:self.max_wav_len]
                else:
                    noise_data = np.resize(noise_data, self.max_wav_len)
                    diff = self.max_wav_len - len(noise_data)
                    # noise_data = np.concatenate(
                    #                             (noise_data, np.zeros((diff,))),
                    #                             axis=0,
                    #                             )
            else:
                noise_data = np.random.randn(self.max_wav_len,)
        else:
            noise_data = np.random.randn(self.max_wav_len,)
        
        signal_energy = np.sum(clean_data**2)
        noise_energy = np.sum(noise_data**2)
        alpha = self.compute_scale(signal_energy, noise_energy, sample_snr_db)
        return clean_data, clean_data + alpha*noise_data


    def __getitem__(self, index):
        target, noisy = self.mix_noise(index)
        target_stft = self.extract_stft_feats(target)
        noisy_stft = self.extract_stft_feats(noisy)
        return noisy_stft, target_stft, noisy, target


    def __len__(self):
        return len(self.utterance_paths)

if __name__ == "__main__":
    hparams = create_hparams()

    dataloader = OnTheFlyMixer(
                                utterance_paths_file=hparams.validation_files,
                                noise_paths_file=hparams.noise_files,
                                hparams=hparams,
                                )

    src_stft, tar_stft, src, tar = dataloader[99]
    
    pylab.figure()
    pylab.imshow(np.log10(src_stft[:,:,0]**2 + src_stft[:,:,1]**2))
    pylab.figure()
    pylab.imshow(np.log10(tar_stft[:,:,0]**2 + tar_stft[:,:,1]**2))




















        
