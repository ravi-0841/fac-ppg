#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 17:10:11 2023

@author: ravi
"""

import librosa
import numpy as np
import torch
import pylab

# from src.common.wsolatsm import wsola
from wsolatsm import wsola
from utils import get_mask_blocks_inference

# Interpolation happens with ceil operation
class WSOLAInterpolationBlockEnergy():
    def __init__(self, win_size=320, hop_size=160, tolerance=160, thresh=1e-3):
        self.win_size = win_size
        self.hop_size = hop_size
        self.tolerance = tolerance
        self.wsola_func = wsola
        self.thresh = thresh

    def __create_new_chunks__(self, chunks, rates):
        gaps = [0]
        
        for i in range(1, len(chunks)):
            cur_chunk, prev_chunk = chunks[i], chunks[i-1]
            gaps.append(cur_chunk[0] - prev_chunk[1] - 1)
        
        new_chunks = []
        for i in range(len(chunks)):
            cur_chunk = chunks[i]
            new_start = cur_chunk[0] + gap[i]
            new_len = 
            
            
    
    def __create_tsf__(self, mask, rates, chunks):
        length_mod_mask = mask[:chunks[0][0]]
        x = librosa.frames_to_samples(np.arange(0, len(mask)), 
                                      hop_length=self.hop_size)
        y = np.ones((len(mask),))
        # y[0] = 0
        for r, c in zip(rates, chunks):
            y[c[0]:c[1]+1] = r
            length_mod_mask = np.concatenate((length_mod_mask, 
                                              np.ones((int(np.ceil(c[2]*r)),))))

        y = librosa.frames_to_samples(np.cumsum(y), 
                                      hop_length=self.hop_size)
        
        samp_points = np.vstack((x.reshape(1,-1), y.reshape(1,-1)))
        return samp_points, length_mod_mask
    
    def __call__(self, mask, rates, speech, chunks):
        # mask -> [1, #Time]
        # rate -> [scalar] 0.7 -> 1.3 in increments of 0.1
        # x -> [1, 1, audio_wav]
        mask = mask.detach().squeeze().cpu().numpy()
        rates = list(rates.cpu().numpy())
        # print("Rates in interpolation: ", rates)
        speech = speech.detach().squeeze().cpu().numpy()
        
        samp_points, updated_mask = self.__create_tsf__(mask, rates, chunks)
        speech_mod = self.wsola_func(x=speech, 
                                     s=samp_points,
                                     win_size=self.win_size,
                                     syn_hop_size=self.hop_size,
                                     tolerance=self.tolerance,
                                     )
        energy_mod = librosa.feature.rms(y=speech_mod, 
                                      frame_length=self.win_size,
                                      hop_length=self.hop_size,
                                      center=True)
        energy_mod = energy_mod.reshape(-1,)
        energy_mask = np.zeros((len(energy_mod),))
        energy_mask[np.where(energy_mod>self.thresh)[0]] = 1
        idx = np.where(energy_mask==1)[0]
        energy_mask[idx[0]:idx[-1]] = 1
        energy_mask = np.multiply(energy_mask, energy_mod)
        
        speech_modified = torch.from_numpy(speech_mod.reshape(1,1,-1)).float()
        energy_modified = torch.from_numpy(energy_mask.reshape(1,1,-1)).float() #energy_mask
        return speech_modified, energy_modified, samp_points, updated_mask


class BatchWSOLAInterpolationEnergy():
    def __init__(self, win_size=320, hop_size=160, tolerance=160, thresh=1e-3):
        self.win_size = win_size
        self.hop_size = hop_size
        self.tolerance = tolerance
        self.wsola_func = wsola
        self.thresh = thresh
    
    def __create_tsf__(self, mask, rate):
        x = librosa.frames_to_samples(np.arange(0, len(mask)), 
                                      hop_length=self.hop_size)
        y = np.ones((len(mask),))
        y[0] = 0
        y[np.where(mask>0)[0]] = rate
        y = librosa.frames_to_samples(np.cumsum(y), 
                                      hop_length=self.hop_size)
        
        samp_points = np.vstack((x.reshape(1,-1), y.reshape(1,-1)))
        return samp_points
    
    def __collate__(self, batch_data):
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x.shape[0] for x in batch_data]), dim=0,
            descending=True)
        max_input_len = input_lengths[0]

        data_padded = torch.FloatTensor(len(batch_data), 1, max_input_len)
        data_padded.zero_()
        
        for i in range(len(ids_sorted_decreasing)):
            curr_speech = batch_data[ids_sorted_decreasing[i]]

            data_padded[i, :, :curr_speech.shape[0]] = curr_speech
            data_padded[i, :, curr_speech.shape[0]:] = 0

        return data_padded
    
    def __call__(self, mask, rate, speech):
        # mask -> [batch, #Time]
        # rate -> [batch, 1] 0.7 -> 1.3 in increments of 0.1
        # x -> [batch, 1, audio_wav]
        
        num_samples = mask.size(0)
        
        batch_mask = mask.detach().squeeze().cpu().numpy()
        batch_rate = rate.detach().squeeze().cpu().numpy()
        batch_speech = speech.detach().squeeze().cpu().numpy()

        batch_mod_speech = []
        batch_mod_energy = []
        batch_samp_points = []

        for n in range(num_samples):
            mask = batch_mask[n]
            rate = batch_rate[n]
            speech = batch_speech[n]
            
            samp_points = self.__create_tsf__(mask, rate)
            batch_samp_points.append(samp_points)
            speech_mod = self.wsola_func(x=speech, 
                                         s=samp_points,
                                         win_size=self.win_size,
                                         syn_hop_size=self.hop_size,
                                         tolerance=self.tolerance,
                                         )
            
            energy_mod = librosa.feature.rms(y=speech_mod, 
                                          frame_length=self.win_size,
                                          hop_length=self.hop_size,
                                          center=True)
            energy_mod = energy_mod.reshape(-1,)
            energy_mask = np.zeros((len(energy_mod),))
            energy_mask[np.where(energy_mod>self.thresh)[0]] = 1
            idx = np.where(energy_mask==1)[0]
            energy_mask[idx[0]:idx[-1]] = 1
            energy_mask = np.multiply(energy_mask, energy_mod)
            
            # speech_modified = torch.from_numpy(speech_modified.reshape(1,1,-1)).float()
            batch_mod_speech.append(torch.from_numpy(speech_mod))
            batch_mod_energy.append(torch.from_numpy(energy_mask))
        
        speech_modified_padded = self.__collate__(batch_mod_speech)
        energy_modified_padded = self.__collate__(batch_mod_energy)
        return speech_modified_padded, energy_modified_padded, batch_samp_points


if __name__ == "__main__":
    dummy_mask = np.concatenate((np.zeros((1,30)), np.ones((1,30)), 
                                  np.zeros((1,20)), np.ones((1,50)), 
                                  np.zeros((1,70))), axis=1)
    dummy_mask = np.tile(np.expand_dims(dummy_mask, axis=-1), reps=(1,1,512))
    # dummy_mask = np.concatenate((dummy_mask, np.concatenate((np.ones((1,30)), 
    #                                                         np.zeros((1,30)), 
    #                                                         np.ones((1,20)), 
    #                                                         np.zeros((1,50)), 
    #                                                         np.ones((1,70))), 
    #                                                         axis=1)), axis=0)
    dummy_mask = torch.from_numpy(dummy_mask)
    dummy_mask_chunks, chunks = get_mask_blocks_inference(dummy_mask)

    dummy_rate = torch.Tensor([0.5, 1.5])
    dummy_speech = torch.from_numpy(np.random.randn(1, 1, 32000))
    wsola_obj = WSOLAInterpolationBlockEnergy()
    (dummy_speech_modified, dummy_energy_modified, 
     samp_points, dummy_mask_modified) = wsola_obj(mask=dummy_mask[:,:,0],
                                                   rates=dummy_rate,
                                                   speech=dummy_speech,
                                                   chunks=chunks
                                                   )
    
    # pylab.figure(), pylab.plot(samp_points[0][0], samp_points[0][1])
    # pylab.figure(), pylab.plot(samp_points[1][0], samp_points[1][1])













































