#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 16:14:38 2023

@author: ravi
"""


# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA Corporation
# Copyright (c) 2019, Guanlong Zhao
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

"""Modified from https://github.com/NVIDIA/tacotron2"""


class HParamsView(object):
    def __init__(self, d):
        self.__dict__ = d


def create_hparams(**kwargs):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = {
        ################################
        # Experiment Parameters        #
        ################################
        "epochs": 100,
        "iters_per_checkpoint": 1000,
        "seed": 1107,
        "dynamic_loss_scaling": True,
        "fp16_run": False,
        "distributed_run": False,
        "dist_backend": "nccl",
        "dist_url": "tcp://localhost:54321",
        "cudnn_enabled": True,
        "cudnn_benchmark": False,
        "output_directory": "./masked_predictor_output",  # Directory to save checkpoints.
        # Directory to save tensorboard logs. Just keep it like this.
        "log_directory": './logs',
        "extended_desc": "AllEmo_minibatch_consistent", # extra description for logging/identification
        "checkpoint_path": "",  # Path to a checkpoint file.
        # "checkpoint_path_inference": "./masked_predictor_output/temp_1_neg_salience_postRate_Angry_TD_RL_1e-05_10.0_2e-07_5.0_batchsize_8/checkpoint_35000",
        "checkpoint_path_inference": "./masked_predictor_output/lr_opposing_1e-05_10.0_2e-07_5.0_AllEmo_minibatch_consistent/checkpoint_145000",  # Path to a checkpoint file.
        # "checkpoint_path_inference": "./masked_predictor_output/lr_opposing_1e-05_10.0_2e-07_5.0_AllEmo/checkpoint_51000",  # Path to a checkpoint file.
        "warm_start": False,  # Load the model only (warm start)
        "n_gpus": 1,  # Number of GPUs
        "rank": 0,  # Rank of current gpu
        "group_name": 'group_name',  # Distributed group name

        ################################
        # Data Parameters             #
        ################################
        # Passed as a txt file, see data/filelists/training-set.txt for an
        # example.
        "complete_files": "./speechbrain_data/VESUS_saliency_complete.txt",
        "training_files": './speechbrain_data/VESUS_saliency_training_small.txt',
        "validation_files": './speechbrain_data/VESUS_saliency_validation_big.txt',
        "testing_files": "./speechbrain_data/VESUS_saliency_testing_big.txt",
        "is_full_ppg": True,  # Whether to use the full PPG or not.
        "is_append_f0": False,  # Currently only effective at sentence level
        "ppg_subsampling_factor": 1,  # Sub-sample the ppg & acoustic sequence.
        # Cases
        # |'load_feats_from_disk'|'is_cache_feats'|Note
        # |True                  |True            |Error
        # |True                  |False           |Please set cache path
        # |False                 |True            |Overwrite the cache path
        # |False                 |False           |Ignores the cache path
        "load_feats_from_disk": False,  # Remember to set the path.
        # Mutually exclusive with 'load_feats_from_disk', will overwrite
        # 'feats_cache_path' if set.
        "is_cache_feats": False,
        "feats_cache_path": '',

        ################################
        # Audio Parameters             #
        ################################
        "max_wav_value": 32768.0,
        "sampling_rate": 16000,
        "downsampling_factor": 160,
        "n_acoustic_feat_dims": 40, #80
        "filter_length": 400, #1024
        "n_fft": 512,
        "hop_length": 160,
        "win_length": 320, #1024
        "mel_fmin": 0.0,
        "mel_fmax": 8000.0,

        ################################
        # Optimization Hyperparameters #
        ################################
        "lambda_prior_KL": 1e-5, # 5e-4
        "lambda_predict": 10.0, # 10
        "lambda_sparse_KL": 2e-07, # 1e-07
        "temp_scale": 5.0, #15.0
        "use_saved_learning_rate": False,
        "learning_rate_saliency": 1e-4, #1e-5
        "learning_rate_rate": 1e-6,
        "learning_rate_decay": 0.955,
        "learning_rate_lb": 1e-7,
        "weight_decay": 1e-6,
        "grad_clip_thresh": 1.0,
        "batch_size": 8, #8
        "mask_padding": True, # set model's padded outputs to padded values
        "alpha": 0.1, # Bernoulli parameter for sampling 1st entry of the mask
        "beta": 0.95, # Bernoulli parameter for mask persistence
    }

    for key, val in kwargs.items():
        if key in hparams:
            hparams[key] = val
        else:
            raise ValueError('The hyper-parameter %s is not supported.' % key)

    hparams_view = HParamsView(hparams)

    return hparams_view


