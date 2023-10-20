#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 14:55:32 2023

@author: ravi
"""


import parselmouth
import librosa
import numpy as np
import soundfile as sf
from parselmouth import praat

sound = parselmouth.Sound("/home/ravi/Desktop/Fold1_1011_DFA_ANG_XX.wav")
f0min=50
f0max=500
pointProcess = praat.call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
formants = praat.call(sound, "To Formant (burg)", 0.001, 5, 5000, 0.01, 99)

data, sr = sf.read("/home/ravi/Desktop/Fold1_1011_DFA_ANG_XX.wav")
spect = librosa.stft(data, n_fft=1024, hop_length=160, win_length=320)
times = librosa.frames_to_time(np.arange(spect.shape[1]), sr=sr, hop_length=160, n_fft=1024)
import numpy as np
times = librosa.frames_to_time(np.arange(spect.shape[1]), sr=sr, hop_length=160, n_fft=1024)
f1_array = []
f2_array = []
f3_array = []

for t in times:
    f1 = praat.call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
    f1_array.append(f1)
    f2 = praat.call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
    f2_array.append(f2)
    f3 = praat.call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
    f3_array.append(f3)