#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 14:51:48 2023

@author: ravi
"""

import numpy as np
import torch
import pyworld as pw

from src.common.feat_utils import smooth, generate_interpolation
from src.common.tdpsolatm import tdpsola
from scipy import ndimage


class PitchModification():
    def __init__(self, frame_period=10):
        self.frame_period = frame_period
    
    def __call__(self, factor, speech):
        # rate -> [scalar] 0.7 -> 1.3 in increments of 0.1
        # x -> [1, 1, audio_wav]
        factor = factor.cpu().numpy()
        speech = speech.detach().squeeze().cpu().numpy()
        
        f0, sp, ap = pw.wav2world(np.asarray(speech, np.float64), 16000, 
                                  frame_period=self.frame_period)
        f0_modified = np.ascontiguousarray(factor * f0)
        speech_modified = pw.synthesize(f0_modified, sp, ap, 16000, 
                                        frame_period=self.frame_period)
        speech_modified = torch.from_numpy(speech_modified.reshape(1,1,-1)).float()
        return speech_modified
    

class BatchPitchModification():
    def __init__(self, frame_period=10):
        self.frame_period = frame_period
    
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
    
    def __call__(self, factor, speech):
        # rate -> [batch, 1] 0.7 -> 1.3 in increments of 0.1
        # x -> [batch, 1, audio_wav]
        
        num_samples = factor.size(0)

        batch_factor = factor.detach().squeeze().cpu().numpy()
        batch_speech = speech.detach().squeeze().cpu().numpy()

        batch_mod_speech = []

        for n in range(num_samples):
            factor = batch_factor[n]
            speech = batch_speech[n]
            speech = speech.reshape((-1,))
            
            f0, sp, ap = pw.wav2world(np.asarray(speech, np.float64), 16000, 
                                      frame_period=self.frame_period)
            f0_modified = np.ascontiguousarray(factor * f0)
            speech_modified = pw.synthesize(f0_modified, sp, ap, 16000, 
                                            frame_period=self.frame_period)
            speech_modified = torch.from_numpy(speech_modified).float()
            batch_mod_speech.append(speech_modified)
        
        speech_modified_padded = self.__collate__(batch_mod_speech)
        return speech_modified_padded


#%%
def local_modification(f0, mask, factor):
    zeros_idx = np.where(f0<=1)[0]
    modif_idx = np.where(mask>0)[0]
    y = smooth(generate_interpolation(f0), window_len=13)
    y[modif_idx] *= factor
    y_smooth = smooth(y, window_len=13)
    y_smooth[np.where(y_smooth<50)[0]] = 50
    y_smooth[np.where(y_smooth>600)[0]] = 600
    y_smooth[zeros_idx] = 0.
    return y_smooth


def local_chunk_modification(f0, chunks, factors):
    zeros_idx = np.where(f0<=1)[0]
    y = smooth(generate_interpolation(f0), window_len=13)
    for i in range(min(len(factors), len(chunks))):
        f = factors[i]
        c = chunks[i]
        try:
            y[c[0]:c[1]+1] *= f
        except Exception as ex:
            pass
    
    y_smooth = smooth(y, window_len=13)
    y_smooth[np.where(y_smooth<50)[0]] = 50
    y_smooth[np.where(y_smooth>600)[0]] = 600
    y_smooth[zeros_idx] = 0.
    return y_smooth
    

class LocalPitchModification():
    def __init__(self, frame_period=10):
        self.frame_period = frame_period
    
    def __call__(self, factors, speech, chunks):
        # rate -> [scalar] 0.5 -> 1.5 in increments of 0.1
        # x -> [1, 1, audio_wav]
        factors = list(factors.cpu().numpy().reshape(-1,))
        speech = speech.detach().squeeze().cpu().numpy()
        
        f0, sp, ap = pw.wav2world(np.asarray(speech, np.float64), 16000, 
                                  frame_period=self.frame_period)
        f0_modified = np.ascontiguousarray(local_chunk_modification(f0, chunks, factors))
        speech_modified = pw.synthesize(f0_modified, sp, ap, 16000, 
                                        frame_period=self.frame_period)
        speech_modified = torch.from_numpy(speech_modified.reshape(1,1,-1)).float()
        return speech_modified


class BatchLocalPitchModification():
    def __init__(self, frame_period=10):
        self.frame_period = frame_period
    
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
    
    def __call__(self, mask, factor, speech):
        # rate -> [batch, 1] 0.7 -> 1.3 in increments of 0.1
        # x -> [batch, 1, audio_wav]
        # mask -> [batch, Time]
        
        num_samples = factor.size(0)

        batch_factor = factor.detach().squeeze().cpu().numpy()
        batch_speech = speech.detach().squeeze().cpu().numpy()

        batch_mod_speech = []

        for n in range(num_samples):
            factor = batch_factor[n]
            speech = batch_speech[n]
            speech = speech.reshape((-1,))
            m = mask.detach().cpu().numpy()
            m = m[n,:].reshape(-1,)
            
            f0, sp, ap = pw.wav2world(np.asarray(speech, np.float64), 16000, 
                                      frame_period=self.frame_period)
            f0_modified = np.ascontiguousarray(local_modification(f0, m, factor))
            speech_modified = pw.synthesize(f0_modified, sp, ap, 16000, 
                                            frame_period=self.frame_period)
            speech_modified = torch.from_numpy(speech_modified).float()
            batch_mod_speech.append(speech_modified)
        
        speech_modified_padded = self.__collate__(batch_mod_speech)
        return speech_modified_padded



#%% TD-PSOLA based pitch modification

def local_modificationTDPSOLA(f0, mask, factor, speech, 
                              sr, hop_size=160, win_size=160):
    aux = np.ones((len(f0),))
    modif_idx = np.where(mask>0)[0]
    aux[modif_idx] = factor
    aux = ndimage.gaussian_filter1d(aux, sigma=3)
    f0_target = f0 * aux
    f0_target = np.minimum(f0_target, 600)
    f0_target = np.maximum(f0_target, 70)
    mod_speech = tdpsola(x=speech.reshape(1,-1), sr=sr, 
                         src_f0=f0, tgt_f0=f0_target, 
                         p_hop_size=hop_size, p_win_size=win_size)
    
    return mod_speech


def local_chunk_modificationTDPSOLA(f0, chunks, factors, speech, sr, 
                                    hop_size=160, win_size=160):
    aux = np.ones((len(f0),))
    for i in range(min(len(factors), len(chunks))):
        f = factors[i]
        c = chunks[i]
        try:
            aux[c[0]:c[1]+1] *= f
        except Exception as ex:
            pass
    
    aux = ndimage.gaussian_filter1d(aux, sigma=3)
    f0_target = f0 * aux
    f0_target = np.minimum(f0_target, 600)
    f0_target = np.maximum(f0_target, 70)
    mod_speech = tdpsola(x=speech.reshape(1,-1), sr=sr, 
                         src_f0=f0, tgt_f0=f0_target, 
                         p_hop_size=hop_size, p_win_size=win_size)

    return mod_speech


class LocalPitchModification_TDPSOLA():
    def __init__(self, sr=16000, frame_period=10):
        self.sr = sr
        self.frame_period = frame_period
    
    def __call__(self, factors, speech, chunks):
        # rate -> [scalar] 0.5 -> 1.5 in increments of 0.1
        # x -> [1, 1, audio_wav]
        factors = list(factors.cpu().numpy().reshape(-1,))
        speech = speech.detach().squeeze().cpu().numpy()
        speech = np.asarray(speech, np.float64)
        f0 = pw.dio(speech, self.sr, frame_period=self.frame_period)[0]
        speech_modified = local_chunk_modificationTDPSOLA(
                                    f0, 
                                    chunks, 
                                    factors,
                                    speech,
                                    sr=self.sr,
                                    hop_size=int(self.sr*self.frame_period/1000),
                                    win_size=int(self.sr*self.frame_period/1000),
                                )
        speech_modified = torch.from_numpy(speech_modified.reshape(1,1,-1)).float()
        return speech_modified


class BatchLocalPitchModification_TDPSOLA():
    def __init__(self, sr=16000, frame_period=10):
        self.sr = sr
        self.frame_period = frame_period
    
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
    
    def __call__(self, mask, factor, speech):
        # rate -> [batch, 1] 0.7 -> 1.3 in increments of 0.1
        # x -> [batch, 1, audio_wav]
        # mask -> [batch, Time]
        
        num_samples = factor.size(0)

        batch_factor = factor.detach().squeeze().cpu().numpy()
        batch_speech = speech.detach().squeeze().cpu().numpy()

        batch_mod_speech = []

        for n in range(num_samples):
            factor = batch_factor[n]
            speech = batch_speech[n]
            speech = speech.reshape((-1,))
            m = mask.detach().cpu().numpy()
            m = m[n,:].reshape(-1,)
            
            speech = np.asarray(speech, np.float64)
            f0 = pw.dio(speech, self.sr, frame_period=self.frame_period)[0]
            speech_modified = local_modificationTDPSOLA(
                                    f0, 
                                    m, 
                                    factor, 
                                    speech, 
                                    sr=self.sr,
                                    hop_size=int(self.sr*self.frame_period/1000),
                                    win_size=int(self.sr*self.frame_period/1000),
                                )
            speech_modified = torch.from_numpy(speech_modified).float()
            batch_mod_speech.append(speech_modified)
        
        speech_modified_padded = self.__collate__(batch_mod_speech)
        return speech_modified_padded




































