#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 17:05:19 2022

@author: ravi
"""

# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""From https://github.com/NVIDIA/tacotron2"""

import random
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from src.common.plotting_utils import plot_alignment_to_numpy, \
    plot_spectrogram_to_numpy


class Wav2VecLogger(SummaryWriter):
    def __init__(self, logdir):
        super(Wav2VecLogger, self).__init__(logdir)

    def log_training(self, reduced_loss, grad_norm, learning_rate, duration,
                     iteration):
        self.add_scalar("training.loss", reduced_loss, iteration)
        self.add_scalar("grad.norm", grad_norm, iteration)
        self.add_scalar("learning.rate", learning_rate, iteration)

    def log_validation(self, reduced_loss, model, x, y, y_pred, iteration):
        self.add_scalar("validation.loss", reduced_loss, iteration)
        ssl_outputs = y_pred.permute(1,2,0)
        ssl_targets = y.permute(1,2,0)
        
        ssl_outputs = torch.log10(torch.abs(ssl_outputs))
        ssl_targets = torch.log10(torch.abs(ssl_targets))

        # plot distribution of parameters
        # try:
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)
        # except Exception as ex:
        #     pass

        # plot alignment, mel target and predicted, gate target and predicted
        # idx = random.randint(0, ssl_targets.size(0) - 1)
        # self.add_image(
        #     "ssl_target",
        #     plot_spectrogram_to_numpy(ssl_targets[idx].data.cpu().numpy()),
        #     iteration)
        # self.add_image(
        #     "ssl_predicted",
        #     plot_spectrogram_to_numpy(ssl_outputs[idx].data.cpu().numpy()),
        #     iteration)


class W2V2DecoderLogger(SummaryWriter):
    def __init__(self, logdir):
        super(W2V2DecoderLogger, self).__init__(logdir)

    def log_training(self, reduced_loss, grad_norm, learning_rate, duration,
                     iteration):
        self.add_scalar("training.loss", reduced_loss, iteration)
        self.add_scalar("grad.norm", grad_norm, iteration)
        self.add_scalar("learning.rate", learning_rate, iteration)

    def log_validation(self, reduced_loss, model, x, y, y_pred, iteration):
        self.add_scalar("validation.loss", reduced_loss, iteration)
        mel_outputs = y_pred[:,:,:,0]**2 + y_pred[:,:,:,1]**2
        mel_targets = y[:,:,:,0]**2 + y[:,:,:,1]**2

        mel_outputs = torch.log10(mel_outputs)
        mel_targets = torch.log10(mel_targets)

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, mel_targets.size(0) - 1)

        self.add_figure(
            "mel_target",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            iteration)
        self.add_figure(
            "mel_predicted",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            iteration)



class WaveglowLogger(SummaryWriter):
    def __init__(self, logdir):
        super(WaveglowLogger, self).__init__(logdir)

    def log_training(self, reduced_loss, iteration):
            self.add_scalar("training.loss", reduced_loss, iteration)
