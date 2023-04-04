#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 15:38:43 2023

@author: ravi
"""

import librosa
import numpy as np
import torch
import pylab

from src.common.wsolatsm import wsola
# from wsolatsm import wsola


class WSOLAInterpolation():
    def __init__(self, win_size=320, hop_size=160, tolerance=160):
        self.win_size = win_size
        self.hop_size = hop_size
        self.tolerance = tolerance
        self.wsola_func = wsola
    
    def __create_tsf__(self, ):
        return None
    
    def __call__(self, mask, rate, speech):
        # mask -> [1, #Time]
        # rate -> [scalar] 0.7 -> 1.3 in increments of 0.1
        # x -> [1, 1, audio_wav]
        mask = mask.detach().squeeze().cpu().numpy()
        rate = rate.cpu().numpy()
        speech = speech.detach().squeeze().cpu().numpy()
        
        x = librosa.frames_to_samples(np.arange(0, len(mask)), 
                                      hop_length=self.hop_size)
        y = np.ones((len(mask),))
        y[np.where(mask>0)[0]] = rate
        y = librosa.frames_to_samples(np.cumsum(y), 
                                      hop_length=self.hop_size)
        
        samp_points = np.vstack((x.reshape(1,-1), y.reshape(1,-1)))
        speech_modified = self.wsola_func(x=speech, 
                                         s=samp_points,
                                         win_size=self.win_size,
                                         syn_hop_size=self.hop_size,
                                         tolerance=self.tolerance,
                                         )
        speech_modified = torch.from_numpy(speech_modified.reshape(1,1,-1)).float()
        return speech_modified, samp_points


class WSOLAInterpolationEnergy():
    def __init__(self, win_size=320, hop_size=160, tolerance=160):
        self.win_size = win_size
        self.hop_size = hop_size
        self.tolerance = tolerance
        self.wsola_func = wsola
    
    def __create_tsf__(self, ):
        return None
    
    def __call__(self, mask, rate, speech):
        # mask -> [1, #Time]
        # rate -> [scalar] 0.7 -> 1.3 in increments of 0.1
        # x -> [1, 1, audio_wav]
        mask = mask.detach().squeeze().cpu().numpy()
        rate = rate.cpu().numpy()
        speech = speech.detach().squeeze().cpu().numpy()
        
        x = librosa.frames_to_samples(np.arange(0, len(mask)), 
                                      hop_length=self.hop_size)
        y = np.ones((len(mask),))
        y[np.where(mask>0)[0]] = rate
        y = librosa.frames_to_samples(np.cumsum(y), 
                                      hop_length=self.hop_size)
        
        samp_points = np.vstack((x.reshape(1,-1), y.reshape(1,-1)))
        speech_modified = self.wsola_func(x=speech, 
                                         s=samp_points,
                                         win_size=self.win_size,
                                         syn_hop_size=self.hop_size,
                                         tolerance=self.tolerance,
                                         )
        energy_modified = librosa.feature.rms(y=speech_modified, 
                                              frame_length=self.win_size,
                                              hop_length=self.hop_size,
                                              center=True)
        energy_modified = energy_modified.reshape(-1,)
        energy_mask = np.zeros((len(energy_modified),))
        energy_mask[np.where(energy_modified>1e-3)[0]] = 1
        idx = np.where(energy_mask==1)[0]
        energy_mask[idx[0]:idx[-1]] = 1
        
        speech_modified = torch.from_numpy(speech_modified.reshape(1,1,-1)).float()
        energy_modified = torch.from_numpy(energy_modified.reshape(1,1,-1)).float() #energy_mask
        return speech_modified, energy_modified, samp_points


class BatchWSOLAInterpolation():
    def __init__(self, win_size=320, hop_size=160, tolerance=160):
        self.win_size = win_size
        self.hop_size = hop_size
        self.tolerance = tolerance
        self.wsola_func = wsola
    
    def __create_tsf__(self, mask, rate):
        x = librosa.frames_to_samples(np.arange(0, len(mask)), 
                                      hop_length=self.hop_size)
        y = np.ones((len(mask),))
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

        speech_padded = torch.FloatTensor(len(batch_data), 1, max_input_len)
        speech_padded.zero_()
        
        for i in range(len(ids_sorted_decreasing)):
            curr_speech = batch_data[ids_sorted_decreasing[i]]

            speech_padded[i, :, :curr_speech.shape[0]] = curr_speech
            speech_padded[i, :, curr_speech.shape[0]:] = 0

        return speech_padded
    
    def __call__(self, mask, rate, speech):
        # mask -> [batch, #Time]
        # rate -> [batch, 1] 0.7 -> 1.3 in increments of 0.1
        # x -> [batch, 1, audio_wav]
        
        num_samples = mask.size(0)
        
        batch_mask = mask.detach().squeeze().cpu().numpy()
        batch_rate = rate.detach().squeeze().cpu().numpy()
        batch_speech = speech.detach().squeeze().cpu().numpy()

        batch_mod_speech = []
        batch_samp_points = []

        for n in range(num_samples):
            mask = batch_mask[n]
            rate = batch_rate[n]
            speech = batch_speech[n]
            
            samp_points = self.__create_tsf__(mask, rate)
            batch_samp_points.append(samp_points)
            speech_modified = self.wsola_func(x=speech, 
                                             s=samp_points,
                                             win_size=self.win_size,
                                             syn_hop_size=self.hop_size,
                                             tolerance=self.tolerance,
                                             )
            # speech_modified = torch.from_numpy(speech_modified.reshape(1,1,-1)).float()
            batch_mod_speech.append(torch.from_numpy(speech_modified))
        
        speech_modified_padded = self.__collate__(batch_mod_speech)
        return speech_modified_padded, batch_samp_points
    

class BatchWSOLAInterpolationEnergy():
    def __init__(self, win_size=320, hop_size=160, tolerance=160):
        self.win_size = win_size
        self.hop_size = hop_size
        self.tolerance = tolerance
        self.wsola_func = wsola
    
    def __create_tsf__(self, mask, rate):
        x = librosa.frames_to_samples(np.arange(0, len(mask)), 
                                      hop_length=self.hop_size)
        y = np.ones((len(mask),))
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
            speech_modified = self.wsola_func(x=speech, 
                                             s=samp_points,
                                             win_size=self.win_size,
                                             syn_hop_size=self.hop_size,
                                             tolerance=self.tolerance,
                                             )
            
            energy_modified = librosa.feature.rms(y=speech_modified, 
                                                  frame_length=self.win_size,
                                                  hop_length=self.hop_size,
                                                  center=True)
            energy_modified = energy_modified.reshape(-1,)
            energy_mask = np.zeros((len(energy_modified),))
            energy_mask[np.where(energy_modified>1e-3)[0]] = 1
            idx = np.where(energy_mask==1)[0]
            energy_mask[idx[0]:idx[-1]] = 1
            
            # speech_modified = torch.from_numpy(speech_modified.reshape(1,1,-1)).float()
            batch_mod_speech.append(torch.from_numpy(speech_modified))
            batch_mod_energy.append(torch.from_numpy(energy_mask))
        
        speech_modified_padded = self.__collate__(batch_mod_speech)
        energy_modified_padded = self.__collate__(batch_mod_energy)
        return speech_modified_padded, energy_modified_padded, batch_samp_points




if __name__ == "__main__":
    dummy_mask = np.concatenate((np.zeros((1,30)), np.ones((1,30)), 
                                 np.zeros((1,20)), np.ones((1,50)), 
                                 np.zeros((1,70))), axis=1)
    dummy_mask = np.concatenate((dummy_mask, np.concatenate((np.ones((1,30)), 
                                                            np.zeros((1,30)), 
                                                            np.ones((1,20)), 
                                                            np.zeros((1,50)), 
                                                            np.ones((1,70))), 
                                                           axis=1)), axis=0)
    dummy_mask = torch.from_numpy(dummy_mask)
    dummy_rate = torch.tensor([[0.5, 1.5]])
    dummy_speech = torch.from_numpy(np.random.randn(2, 1, 32000))
    wsola_obj = BatchWSOLAInterpolation()
    dummy_speech_modified, samp_points = wsola_obj(mask=dummy_mask,
                                                   rate=dummy_rate,
                                                   speech=dummy_speech,
                                                   )
    
    pylab.figure(), pylab.plot(samp_points[0][0], samp_points[0][1])
    pylab.figure(), pylab.plot(samp_points[1][0], samp_points[1][1])













































