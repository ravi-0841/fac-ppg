#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 16:11:58 2023

@author: ravi
"""

class HParamsView(object):
    def __init__(self, d):
        self.__dict__ = d


def create_hparams(**kwargs):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = {
        ################################
        # Experiment Parameters        #
        ################################
        "epochs": 500,
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
        "extended_desc": "temp_0.2_RE_L1_loss", # extra description for logging/identification
        "checkpoint_path": "",  # Path to a checkpoint file.
        "checkpoint_path_saliency": "./masked_predictor_output/vesus_1e-05_10.0_0.0002_5.0_BiLSTM_maskGen_evm_wsola_aug/checkpoint_78000",
        "checkpoint_path_rate": "",#"./masked_predictor_output/VESUS_PitchRate_entropy_0.035_exploit_0.15_temp_0.2_RE_L1_loss/checkpoint_233000",
        "checkpoint_path_inference": "./masked_predictor_output/VESUS_Block_Local_PitchRate_entropy_0.05_exploit_0.15_temp_0.2_RE_L1_loss_max_pretdp_directLoss/checkpoint",
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
        "exploitation_prob": 0.15, #0.15,
        "lambda_entropy": 0.05, #0.05,
        "lambda_critic": 5,
        "temp_scale": 5.0, #15.0
        "exploration_decay": 1.03183,
        "use_saved_learning_rate": False,
        "learning_rate_actor": 1e-6,
        "learning_rate_critic": 1e-6,
        "learning_rate_decay": 0.954992586, #0.912011, #0.955
        "learning_rate_lb": 1e-6, #1e-6
        "learning_rate_ub": 1e-5, #1e-5
        "weight_decay": 1e-6, #1e-6
        "grad_clip_thresh": 1.0,
        "batch_size": 4, #4
        "minibatch_consistency": False,
        "mask_padding": True, # set model's padded outputs to padded values
        "alpha": 0.01, # Bernoulli parameter for sampling 1st entry of the mask
        "beta": 0.95, #0.95 Bernoulli parameter for mask persistence
    }

    for key, val in kwargs.items():
        if key in hparams:
            hparams[key] = val
        else:
            raise ValueError('The hyper-parameter %s is not supported.' % key)

    hparams_view = HParamsView(hparams)

    return hparams_view


