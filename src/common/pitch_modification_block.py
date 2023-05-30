#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 14:51:48 2023

@author: ravi
"""

import numpy as np
import torch
import pyworld as pw


class PitchModification():
    def __init__(self, frame_period=10):
        self.frame_period = frame_period
    
    def __call__(self, factor, speech):
        # rate -> [scalar] 0.7 -> 1.3 in increments of 0.1
        # x -> [1, 1, audio_wav]
        factor = factor.cpu().numpy()
        speech = speech.detach().squeeze().cpu().numpy()
        
        f0, sp, ap = pw.wav2world(np.asarray(speech, np.foat64), 16000, 
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






































