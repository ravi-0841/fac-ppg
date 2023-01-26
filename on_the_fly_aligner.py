#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 12:36:48 2022

@author: ravi
"""


import numpy as np
import soundfile as sf
import torch
import pylab
import librosa
import scipy.spatial as scispa

from src.common.hparams_onflyaligner import create_hparams
from src.common.utils import load_filepaths
from torch.utils.data import DataLoader


#%%
class OnTheFlyAligner():

    def __init__(
                    self, 
                    utterance_paths_file,
                    hparams,
                    cutoff_length=3,
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
    
    
    def normalize(self, sig, rms_level=0):
        """
        Normalize the signal given a certain technique (peak or rms).
        Args:
            - data    (np.ndarray) : input signal.
            - rms_level (int) : rms level in dB.
        """
        # linear rms level and scaling factor
        r = 10**(rms_level / 10.0)
        a = np.sqrt( (len(sig) * r**2) / np.sum(sig**2) )
    
        # normalize
        y = sig * a
        return y


    def get_alignment(self, index):
        file_paths = self.utterance_paths[index].split(", ")
        src_path, tar_path = file_paths[0], file_paths[1]

        src_data, sr = sf.read(src_path)
        tar_data, sr = sf.read(tar_path)
        
        # Normalize to RMS value of 0 dB
        # src_data = self.normalize(src_data)
        # tar_data = self.normalize(tar_data)
        
        # Mixing noise in the source signal
        sample_snr_db =  10 + 5*np.random.rand()  # Range is 5-10db
        noise_data = np.random.randn(len(src_data),)
        signal_energy = np.sum(src_data**2)
        noise_energy = np.sum(noise_data**2)
        alpha = self.compute_scale(signal_energy, noise_energy, sample_snr_db)
        mix_data = src_data + alpha*noise_data
        src_noisy_stft = self.extract_stft_feats(mix_data)
        src_stft = self.extract_stft_feats(src_data)
        tar_stft = self.extract_stft_feats(tar_data)
        
        # No noise mixing
        # src_stft = self.extract_stft_feats(src_data)
        # tar_stft = self.extract_stft_feats(tar_data)

        # DTW alignment using STFT        
        # src_dtw_stft = librosa.core.stft(
        #                                 src_data, 
        #                                 n_fft=self.hparams.n_fft,
        #                                 hop_length=self.hparams.hop_length,
        #                                 win_length=self.hparams.win_length,
        #                                 )
        # tar_dtw_stft = librosa.core.stft(
        #                                 tar_data, 
        #                                 n_fft=self.hparams.n_fft,
        #                                 hop_length=self.hparams.hop_length,
        #                                 win_length=self.hparams.win_length,
        #                                 )
        
        # DTW alignment using MFCC features
        src_dtw_mfc = librosa.feature.mfcc(
                                        y=src_data,
                                        sr=sr,
                                        n_mfcc=20,
                                        hop_length=self.hparams.hop_length,
                                        win_length=self.hparams.win_length,
                                        )
        
        tar_dtw_mfc = librosa.feature.mfcc(
                                        y=tar_data,
                                        sr=sr,
                                        n_mfcc=20,
                                        hop_length=self.hparams.hop_length,
                                        win_length=self.hparams.win_length,
                                        )
        
        # src_mag_stft = np.sqrt(src_stft.numpy()[:,:,0]**2 + src_stft.numpy()[:,:,1]**2)
        # tar_mag_stft = np.sqrt(tar_stft.numpy()[:,:,0]**2 + tar_stft.numpy()[:,:,1]**2)

        D, wp = librosa.sequence.dtw(src_dtw_mfc, tar_dtw_mfc, 
                                     metric="cosine")
        
        return D, wp, src_stft, tar_stft, src_noisy_stft


    def __getitem__(self, index):
        cost, alignment, src_stft, tar_stft, src_noisy_stft = self.get_alignment(index)
        ext_src_stft = src_stft[:,alignment[:,0],:]
        ext_src_noisy_stft = src_noisy_stft[:,alignment[:,0],:]
        ext_tar_stft = tar_stft[:,alignment[:,1],:]
        return ext_src_stft, ext_tar_stft, ext_src_noisy_stft, src_stft, tar_stft


    def __len__(self):
        return len(self.utterance_paths)


def acoustics_collate(batch):
    """Zero-pad the acoustic sequences in a mini-batch.

    Args:
        batch: An array with B elements, each is a tuple 
        (ext_src_stft, ext_tar_stft, ext_src_noisy_stft, src_stft, tar_stft).

    Returns:
        src_stft_padded: A (batch_size, feature_dim_1, num_frames_1, num_channels_1)
        tensor.
        tar_stft_padded: A (batch_size, feature_dim_1, num_frames_1, num_channels_1)
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

    src_stft_padded = torch.FloatTensor(len(batch), stft_dim, max_input_len, 2)
    src_noisy_stft_padded = torch.FloatTensor(len(batch), stft_dim, max_input_len, 2)
    tar_stft_padded = torch.FloatTensor(len(batch), stft_dim, max_input_len, 2)
    
    src_stft_padded.zero_()
    src_noisy_stft_padded.zero_()
    tar_stft_padded.zero_()
    
    for i in range(len(ids_sorted_decreasing)):
        curr_src_stft = batch[ids_sorted_decreasing[i]][0]
        curr_tar_stft = batch[ids_sorted_decreasing[i]][1]
        curr_src_noisy_stft = batch[ids_sorted_decreasing[i]][2]

        src_stft_padded[i, :, :curr_src_stft.shape[1], :] = curr_src_stft
        src_stft_padded[i, :, curr_src_stft.shape[1]:, :] = curr_src_stft[:, -2:-1, :]

        src_noisy_stft_padded[i, :, :curr_src_noisy_stft.shape[1], :] = curr_src_noisy_stft
        src_noisy_stft_padded[i, :, curr_src_noisy_stft.shape[1]:, :] = curr_src_noisy_stft[:, -2:-1, :]

        tar_stft_padded[i, :, :curr_tar_stft.shape[1], :] = curr_tar_stft
        tar_stft_padded[i, :, curr_tar_stft.shape[1]:, :] = curr_tar_stft[:, -2:-1, :]

    return src_stft_padded, tar_stft_padded, src_noisy_stft_padded, input_lengths


#%%
if __name__ == "__main__":
    hparams = create_hparams()

    trainset = OnTheFlyAligner(
                                utterance_paths_file="./speechbrain_data/neutral_to_angry_train.txt",
                                hparams=hparams,
                                )
    
    torch_data_loader = DataLoader(trainset, num_workers=4, shuffle=True,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=acoustics_collate)
    
    for i, batch in enumerate(torch_data_loader):
        # break
        pass

    # src_stft, tar_stft = dataloader[np.random.choice(np.arange(0, len(dataloader)))]
    
    # pylab.figure()
    # pylab.imshow(np.log10(src_stft[:,:,0]**2 + src_stft[:,:,1]**2))
    # pylab.figure()
    # pylab.imshow(np.log10(tar_stft[:,:,0]**2 + tar_stft[:,:,1]**2))




















        
