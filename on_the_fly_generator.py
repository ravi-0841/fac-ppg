#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 12:55:13 2023

@author: ravi
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 12:36:48 2022

@author: ravi
"""


import numpy as np
import random
import soundfile as sf
import torch
import pylab
import librosa
import scipy.spatial as scispa

from src.common.hparams_onflyaligner import create_hparams
from src.common.utils import load_filepaths
from torch.utils.data import DataLoader


#%%
class OnTheFlyGenerator():

    def __init__(
                    self,
                    min_length=200,
                    max_length=350,
                    stft_len=257,
                ):
        
        self.max_length = max_length
        self.min_length = min_length
        self.stft_len = stft_len

    def random_partition(self, k, iterable):
        results = [[] for i in range(k)]
        for value in iterable:
            x = random.randrange(k)
            results[x].append(value)
        return results

    def generate_data(self):
        gen_length = np.random.randint(self.min_length, self.max_length)
        num_segments = np.random.randint(20, 30)
        partition = self.random_partition(num_segments, np.arange(gen_length))
        voiced_mean = np.random.randn(self.stft_len, 1)
        voiced = 0
        data = np.empty((self.stft_len,0))
        for i in range(num_segments):
            len_cur_seg = len(partition[i])
            if voiced == 0:
                r = np.repeat(voiced_mean, len_cur_seg, axis=1)
                r += np.random.randn(self.stft_len, len_cur_seg)
                data = np.concatenate((data, np.abs(r)), axis=1)
                voiced = 1
            else:
                r = np.repeat(voiced_mean, len_cur_seg, axis=1)
                r += 0.1 * np.random.randn(self.stft_len, len_cur_seg)
                data = np.concatenate((data, np.abs(r)), axis=1)
                voiced = 0
                voiced_mean = np.random.randn(self.stft_len, 1)
        
        data = data/np.max(data)
        
        return torch.from_numpy(data).float()


    def __getitem__(self, index):
        return self.generate_data()


    def __len__(self):
        return 2000


def acoustics_collate(batch):
    """Zero-pad the acoustic sequences in a mini-batch.

    Args:
        batch: An array with B elements, each is a tensor
        (stft -> Freq x Time).

    Returns:
        stft_padded: A (batch_size, Freq, Time) tensor.
        lengths: A batch_size array, each containing the actual length
        of the input sequence.
    """
    # Right zero-pad all sequences to max input length.
    input_lengths, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([x.shape[1] for x in batch]), dim=0,
        descending=True)
    max_input_len = input_lengths[0]
    stft_dim = batch[0].shape[0]

    stft_padded = torch.FloatTensor(len(batch), stft_dim, max_input_len)
    
    stft_padded.zero_()
    
    for i in range(len(ids_sorted_decreasing)):
        curr_stft = batch[ids_sorted_decreasing[i]]

        stft_padded[i, :, :curr_stft.shape[1]] = curr_stft
        stft_padded[i, :, curr_stft.shape[1]:] = 0

    return stft_padded, input_lengths


#%%
if __name__ == "__main__":
    hparams = create_hparams()

    trainset = OnTheFlyGenerator()
    
    torch_data_loader = DataLoader(trainset, num_workers=4, shuffle=True,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=acoustics_collate)
    
    for i, batch in enumerate(torch_data_loader):
        # break
        pass




















        
