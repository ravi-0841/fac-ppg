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


class WSOLAInterpolation():
    def __init__(self, win_size=320, hop_size=160, tolerance=160):
        self.win_size = win_size
        self.hop_size = hop_size
        self.tolerance = tolerance
        self.wsola_func = wsola
    
    def __create_tsf__(self, ):
        return None
    
    def __call__(self, mask, rate, speech):
        # mask -> [batch, #Time]
        # rate -> [scalar] 0.7 -> 1.3 in increments of 0.1
        # x -> [batch, 1, audio_wav]
        mask = mask.detach().squeeze().cpu().numpy()
        rate = rate.cpu().numpy()
        speech = speech.squeeze().cpu().numpy()
        
        x = librosa.frames_to_samples(np.arange(0, len(mask)), 
                                      hop_length=self.hop_size)
        y = np.ones((len(mask),))
        y[np.where(mask==1)[0]] = rate
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
        return speech_modified, x, y


if __name__ == "__main__":
    dummy_mask = torch.from_numpy(np.concatenate((np.zeros((1,30,1)), np.ones((1,30,1)), 
                                 np.zeros((1,20,1)), np.ones((1,50,1)), 
                                 np.zeros((1,70,1))), axis=1))
    dummy_rate = torch.tensor(1.5)
    dummy_speech = torch.from_numpy(np.random.randn(1,1,32000))
    wsola_obj = WSOLAInterpolation()
    dummy_speech_modified, x, y = wsola_obj(mask=dummy_mask,
                                            rate=dummy_rate,
                                            speech=dummy_speech,
                                        )
    
    pylab.figure(), pylab.plot(x, y)













































