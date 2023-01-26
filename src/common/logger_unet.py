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


class UnetLogger(SummaryWriter):
    def __init__(self, logdir):
        super(UnetLogger, self).__init__(logdir)

    def log_training(self, reduced_loss, grad_norm, learning_rate, duration,
                     iteration):
        self.add_scalar("training.loss", reduced_loss, iteration)
        self.add_scalar("grad.norm", grad_norm, iteration)
        self.add_scalar("learning.rate", learning_rate, iteration)

    def log_validation(self, reduced_loss, model, x, y, y_pred, iteration):
        self.add_scalar("validation.loss", reduced_loss, iteration)
        mel_inputs = x[:,0,:,:]**2 + x[:,1,:,:]**2
        mel_outputs = y_pred[:,0,:,:]**2 + y_pred[:,1,:,:]**2
        mel_targets = y[:,0,:,:]**2 + y[:,1,:,:]**2
        
        mel_inputs = torch.log10(mel_inputs.permute(0,2,1))
        mel_outputs = torch.log10(mel_outputs.permute(0,2,1))
        mel_targets = torch.log10(mel_targets.permute(0,2,1))

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, mel_targets.size(0) - 1)

        self.add_figure(
            "mel_input",
            plot_spectrogram_to_numpy(mel_inputs[idx].data.cpu().numpy()),
            iteration)
        self.add_figure(
            "mel_target",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            iteration)
        self.add_figure(
            "mel_predicted",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            iteration)

        # self.add_image(
        #     "mel_input",
        #     mel_inputs[idx].data.cpu().numpy(),
        #     iteration,
        #     dataformats="HW")
        # self.add_image(
        #     "mel_target",
        #     mel_targets[idx].data.cpu().numpy(),
        #     iteration,
        #     dataformats="HW")
        # self.add_image(
        #     "mel_predicted",
        #     mel_outputs[idx].data.cpu().numpy(),
        #     iteration,
        #     dataformats="HW")


class WaveglowLogger(SummaryWriter):
    def __init__(self, logdir):
        super(WaveglowLogger, self).__init__(logdir)

    def log_training(self, reduced_loss, iteration):
            self.add_scalar("training.loss", reduced_loss, iteration)
