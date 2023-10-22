#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:04:36 2023

@author: ravi
"""

import numpy as np
import torch
import pyworld as pw

from src.common.feat_utils import smooth, generate_interpolation


class PitchEnergyModification():
    def __init__(self, frame_period=10):
        self.frame_period = frame_period
    
    def __call__(self, factor_pitch, factor_energy, speech):
        # rate -> [scalar] 0.7 -> 1.3 in increments of 0.1
        # x -> [1, 1, audio_wav]
        factor_pitch = factor_pitch.cpu().numpy()
        factor_energy = factor_energy.cpu().numpy()
        
        speech = speech.detach().squeeze().cpu().numpy()
        
        f0, sp, ap = pw.wav2world(np.asarray(speech, np.float64), 16000, 
                                  frame_period=self.frame_period)
        f0_modified = np.ascontiguousarray(factor_pitch * f0)
        
        energy_current = np.sum(sp**2, axis=-1, keepdims=True)
        energy_modified = factor_energy * energy_current
        sp_modified = sp * (np.sqrt(energy_modified / (energy_current + 1e-20)))
        sp_modified = np.ascontiguousarray(sp_modified)
        
        speech_modified = pw.synthesize(f0_modified, sp_modified, ap, 16000, 
                                        frame_period=self.frame_period)
        speech_modified = torch.from_numpy(speech_modified.reshape(1,1,-1)).float()
        return speech_modified
    

class BatchPitchEnergyModification():
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
    
    def __call__(self, factor_pitch, factor_energy, speech):
        # rate -> [batch, 1] 0.7 -> 1.3 in increments of 0.1
        # x -> [batch, 1, audio_wav]
        
        num_samples = factor_pitch.size(0)

        batch_factor_pitch = factor_pitch.detach().squeeze().cpu().numpy()
        batch_factor_energy = factor_energy.detach().squeeze().cpu().numpy()
        batch_speech = speech.detach().squeeze().cpu().numpy()

        batch_mod_speech = []

        for n in range(num_samples):
            factor_pitch = batch_factor_pitch[n]
            factor_energy = batch_factor_energy[n]
            speech = batch_speech[n]
            speech = speech.reshape((-1,))
            
            f0, sp, ap = pw.wav2world(np.asarray(speech, np.float64), 16000, 
                                      frame_period=self.frame_period)
            f0_modified = np.ascontiguousarray(factor_pitch * f0)
            
            energy_current = np.sum(sp**2, axis=-1, keepdims=True)
            energy_modified = factor_energy * energy_current
            sp_modified = sp * (np.sqrt(energy_modified / (energy_current + 1e-20)))
            sp_modified = np.ascontiguousarray(sp_modified)
            
            speech_modified = pw.synthesize(f0_modified, sp_modified, ap, 16000, 
                                            frame_period=self.frame_period)
            speech_modified = torch.from_numpy(speech_modified).float()
            batch_mod_speech.append(speech_modified)
        
        speech_modified_padded = self.__collate__(batch_mod_speech)
        return speech_modified_padded


#%%
def local_modification(f0, ec, mask, 
                       factor_pitch, 
                       factor_energy):
    modif_idx = np.where(mask>0)[0]
    
    zeros_idx = np.where(f0<=1)[0]
    y = smooth(generate_interpolation(f0), window_len=13)
    y[modif_idx] *= factor_pitch
    f0_modified = smooth(y, window_len=13)
    f0_modified[np.where(f0_modified<50)[0]] = 50
    f0_modified[np.where(f0_modified>600)[0]] = 600
    f0_modified[zeros_idx] = 0.
    
    z = ec.copy()
    z[modif_idx] *= factor_energy
    ec_modified = np.reshape(smooth(z.reshape(-1,), window_len=7), (-1,1))
    return f0_modified, ec_modified


def local_chunk_modification(f0, ec, chunks, 
                             factors_pitch, 
                             factors_energy):
    zeros_idx = np.where(f0<=1)[0]
    y = smooth(generate_interpolation(f0), window_len=13)
    z = ec.copy()
    for i in range(min(len(factors_pitch), len(chunks))):
        fp = factors_pitch[i]
        fe = factors_energy[i]
        c = chunks[i]
        try:
            y[c[0]:c[1]+1] *= fp
            z[c[0]:c[1]+1] *= fe
        except Exception as ex:
            pass
    
    f0_modified = smooth(y, window_len=13)
    f0_modified[np.where(f0_modified<50)[0]] = 50
    f0_modified[np.where(f0_modified>600)[0]] = 600
    f0_modified[zeros_idx] = 0.
    
    ec_modified = np.reshape(smooth(z.reshape(-1,), window_len=7), (-1,1))
    return f0_modified, ec_modified
    

class LocalPitchEnergyModification():
    def __init__(self, frame_period=10):
        self.frame_period = frame_period
    
    def __call__(self, factors_pitch, factors_energy, speech, chunks):
        # rate -> [scalar] 0.5 -> 1.5 in increments of 0.1
        # x -> [1, 1, audio_wav]
        factors_pitch = list(factors_pitch.cpu().numpy().reshape(-1,))
        factors_energy = list(factors_energy.cpu().numpy().reshape(-1,))
        speech = speech.detach().squeeze().cpu().numpy()
        
        f0, sp, ap = pw.wav2world(np.asarray(speech, np.float64), 16000, 
                                  frame_period=self.frame_period)
        ec = np.sum(sp**2, axis=-1, keepdims=True)
        
        f0_modified, ec_modified = local_chunk_modification(f0, ec, chunks, 
                                                            factors_pitch,
                                                            factors_energy)
        f0_modified = np.ascontiguousarray(f0_modified)

        sp_modified = sp * (np.sqrt(ec_modified / (ec + 1e-20)))
        sp_modified = np.ascontiguousarray(sp_modified)
        
        speech_modified = pw.synthesize(f0_modified, sp_modified, ap, 16000, 
                                        frame_period=self.frame_period)
        speech_modified = torch.from_numpy(speech_modified.reshape(1,1,-1)).float()
        return speech_modified


class BatchLocalPitchEnergyModification():
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
    
    def __call__(self, mask, factor_pitch, 
                 factor_energy, speech):
        # rate -> [batch, 1] 0.7 -> 1.3 in increments of 0.1
        # x -> [batch, 1, audio_wav]
        # mask -> [batch, Time]
        
        num_samples = factor_pitch.size(0)

        batch_factor_pitch = factor_pitch.detach().squeeze().cpu().numpy()
        batch_factor_energy = factor_energy.detach().squeeze().cpu().numpy()
        batch_speech = speech.detach().squeeze().cpu().numpy()

        batch_mod_speech = []

        for n in range(num_samples):
            factor_pitch = batch_factor_pitch[n]
            factor_energy = batch_factor_energy[n]
            speech = batch_speech[n]
            speech = speech.reshape((-1,))
            m = mask.detach().cpu().numpy()
            m = m[n,:].reshape(-1,)
            
            f0, sp, ap = pw.wav2world(np.asarray(speech, np.float64), 16000, 
                                      frame_period=self.frame_period)
            ec = np.sum(sp**2, axis=-1, keepdims=True)
            
            f0_modified, ec_modified = local_modification(f0, ec, m, 
                                                          factor_pitch, 
                                                          factor_energy)
            f0_modified = np.ascontiguousarray(f0_modified)
            
            sp_modified = sp * (np.sqrt(ec_modified / (ec + 1e-20)))
            sp_modified = np.ascontiguousarray(sp_modified)
            
            speech_modified = pw.synthesize(f0_modified, sp_modified, ap, 16000, 
                                            frame_period=self.frame_period)
            speech_modified = torch.from_numpy(speech_modified).float()
            batch_mod_speech.append(speech_modified)
        
        speech_modified_padded = self.__collate__(batch_mod_speech)
        return speech_modified_padded
































