#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:29:46 2023

@author: ravi
"""

import os
import sys
import time
import math
import torch
import numpy as np
from torch.utils.data import DataLoader
from block_pitch_duration_RL_max_2 import MaskedRateModifier, RatePredictor
from on_the_fly_augmentor_raw_voice_mask import OnTheFlyAugmentor, acoustics_collate_raw
from src.common.loss_function import (MaskedSpectrogramL1LossReduced,
                                        ExpectedKLDivergence,
                                        VecExpectedKLDivergence, 
                                        SparsityKLDivergence,
                                        EntropyLoss, 
                                        RateLoss,
                                        PitchRateLoss,
                                        BlockPitchRateLoss,
                                    )
from src.common.logger_PitchRatePred import SaliencyPredictorLogger
from src.common.hparams_onflyenergy_block_pitch_rate_vesus import create_hparams
from src.common.interpolation_block import (WSOLAInterpolation,
                                            WSOLAInterpolationEnergy,
                                            BatchWSOLAInterpolation,
                                            BatchWSOLAInterpolationEnergy)
from src.common.pitch_modification_block import (PitchModification,
                                                 BatchPitchModification,
                                                 LocalPitchModification,
                                                 BatchLocalPitchModification,
                                                 )
from src.common.utils import intended_saliency, get_random_mask_chunk
from pprint import pprint


def batchnorm_to_float(module):
    """Converts batch norm modules to FP32"""
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        batchnorm_to_float(child)
    return module


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = OnTheFlyAugmentor(
                        utterance_paths_file=hparams.training_files,
                        hparams=hparams,
                        augment=True,
                    )
    hparams.load_feats_from_disk = False
    hparams.is_cache_feats = False
    hparams.feats_cache_path = ''
    valset = OnTheFlyAugmentor(
                        utterance_paths_file=hparams.validation_files,
                        hparams=hparams,
                        augment=False,
                    )

    collate_fn = acoustics_collate_raw
    
    train_loader = DataLoader(
                            trainset,
                            num_workers=4,
                            shuffle=True,
                            sampler=None,
                            batch_size=hparams.batch_size,
                            drop_last=True,
                            collate_fn=collate_fn,
                            )
    return train_loader, valset, collate_fn


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = SaliencyPredictorLogger(os.path.join(output_directory, 
                                                        log_directory))
    else:
        logger = None
    return logger


def load_model(hparams):
    model_saliency = MaskedRateModifier(hparams.temp_scale).cuda()
    model_rate = RatePredictor(temp_scale=0.2).cuda()
    return model_saliency, model_rate


def warm_start_model(checkpoint_path, model_rate):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_rate.load_state_dict(checkpoint_dict['state_dict_rate'])
    return model_rate


def load_checkpoint_rate(checkpoint_path, model_rate, optimizer_rate):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_rate.load_state_dict(checkpoint_dict['state_dict_rate'])
    optimizer_rate.load_state_dict(checkpoint_dict['optimizer_rate'])
    learning_rate_rate = checkpoint_dict['learning_rate_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return (model_rate, optimizer_rate, learning_rate_rate, iteration)


def load_checkpoint_saliency(checkpoint_path, model_saliency):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_saliency.load_state_dict(checkpoint_dict['state_dict'])
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint saliency '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model_saliency, iteration


def save_checkpoint(model_rate, optimizer_rate, learning_rate_rate, 
                    iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict_rate': model_rate.state_dict(),
                'optimizer_rate': optimizer_rate.state_dict(),
                'learning_rate_rate': learning_rate_rate}, filepath)


def validate(model_saliency, model_rate, WSOLA, OLA, criterion, valset, 
             collate_fn, iteration, batch_size, rate_classes, 
             consistency, n_gpus, logger, distributed_run, rank):
    """Handles all the validation scoring and printing"""
    model_rate.eval()
    with torch.no_grad():
        val_loader = DataLoader(
                                valset,
                                sampler=None,
                                num_workers=4,
                                shuffle=True,
                                batch_size=batch_size,
                                collate_fn=collate_fn,
                                drop_last=True,
                            )

        val_loss = 0.0
        local_iter = 0
        for i, batch in enumerate(val_loader):
            # try:
            x, em = batch[0].to("cuda"), batch[1].to("cuda")
            intent, cats = intended_saliency(batch_size=batch_size, 
                                             consistent=consistency)
            feats, posterior, mask_sample, orig_pred = model_saliency(x, em)
            mask_sample = get_random_mask_chunk(mask_sample)

            (rate_distribution,
             pitch_distribution) = model_rate(feats, mask_sample, intent)
            index_rate = torch.argmax(rate_distribution, dim=-1)
            index_pitch = torch.argmax(pitch_distribution, dim=-1)
            
            # rate = 0.5 + 0.1*index_rate # 0.2*index
            # pitch = 0.5 + 0.1*index_pitch # 0.2*index
            
            rate = 0.25 + 0.15*index_rate # 0.2*index
            pitch = 0.25 + 0.15*index_pitch # 0.2*index
            # pitch = 0.5 + 0.1*index_pitch
            
            dur_mod_speech = OLA(mask=mask_sample[:,:,0], 
                                 factor=pitch, speech=x)
            mod_speech, mod_e, _ = WSOLA(mask=mask_sample[:,:,0], 
                                         rate=rate, speech=dur_mod_speech)
            mod_speech = mod_speech.to("cuda")
            mod_e = mod_e.to("cuda")
            _, _, _, y_pred = model_saliency(mod_speech, mod_e)
            
            ## direct score maximization
            # intent_indices = torch.argmax(intent, dim=-1)
            loss_rate = 1 - y_pred.gather(1,cats.view(-1,1)).view(-1)
            
            ## minimizing a target saliency distribution
            # loss_rate = torch.sum(torch.abs(y_pred - intent), dim=-1)
            
            # corresp_probs = rate_distribution.gather(1,index.view(-1,1)).view(-1)
            # loss_rate = torch.mean(torch.mul(loss_rate, torch.log(corresp_probs)))
            reduced_val_loss = torch.mean(loss_rate).item()
            val_loss += reduced_val_loss
            local_iter += 1
            # except Exception as ex:
            #     print(ex)
            
        val_loss = val_loss / local_iter

    model_rate.train()
    if rank == 0:
        print("Validation loss {}: {:9f}  ".format(iteration, val_loss))
        logger.log_validation(
                                val_loss,
                                model_saliency,
                                model_rate,
                                x,
                                intent,
                                y_pred - orig_pred,
                                posterior[:,:,1:2],
                                mask_sample[:,:,0:1],
                                rate_distribution,
                                pitch_distribution,
                                rate_classes,
                                iteration,
                            )
        # logger_rate.log_parameters(model_rate, iteration)


def train(output_directory, log_directory, checkpoint_path_rate,
          checkpoint_path_saliency, warm_start, n_gpus, rank, 
          group_name, hparams):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model_saliency, model_rate = load_model(hparams)
    learning_rate_rate = hparams.learning_rate_rate

    optimizer_rate = torch.optim.Adam(model_rate.parameters(), 
                                      lr=learning_rate_rate, 
                                      weight_decay=hparams.weight_decay)
    
    criterion1 = torch.nn.L1Loss()
    criterion2 = EntropyLoss()
    criterion3 = BlockPitchRateLoss()

    logger = prepare_directories_and_logger(output_directory, log_directory, rank)

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    # rate_classes = [str(np.round(x,2)) for x in np.arange(0.5, 1.6, 0.1)]
    rate_classes = [str(np.round(x,2)) for x in np.arange(0.25, 1.9, 0.15)]
    

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    
    assert checkpoint_path_saliency != "", "Need saliency model for feedback"
    
    model_saliency, _ = load_checkpoint_saliency(checkpoint_path_saliency,
                                                 model_saliency,
                                                 )
    
    if checkpoint_path_rate:
        if warm_start:
            model_rate = warm_start_model(checkpoint_path_rate, model_rate)
        else:
            (
                model_rate,
                optimizer_rate,
                _learning_rate_rate,
                iteration,
            ) = load_checkpoint_rate(
                                    checkpoint_path_rate,
                                    model_rate,
                                    optimizer_rate,
                                    )
            if hparams.use_saved_learning_rate:
                learning_rate_rate = _learning_rate_rate

            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    num_params = sum(p.numel() for p in model_rate.parameters() if p.requires_grad)
    print("Total number of trainable parameters are: ", num_params)
    
    WSOLA = BatchWSOLAInterpolationEnergy(win_size=hparams.win_length, 
                                   hop_size=hparams.hop_length,
                                   tolerance=hparams.hop_length,
                                   thresh=1e-3)
    OLA = BatchLocalPitchModification(frame_period=10)

    model_saliency.eval()
    model_rate.train()
    # ================ MAIN TRAINING LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            # try:
            start = time.perf_counter()
            for param_group in optimizer_rate.param_groups:
                param_group['lr'] = learning_rate_rate

            model_rate.zero_grad()

            (x, e, l) = (batch[0].to("cuda"), batch[1].to("cuda"),
                          batch[3])
            l = torch.div(l, hparams.downsampling_factor, 
                          rounding_mode="floor")

            # input_shape should be [#batch_size, 1, #time]
            feats, posterior, mask_sample, y_pred = model_saliency(x, e)
            mask_sample = get_random_mask_chunk(mask_sample)
            
            # Intended Saliency
            intent_saliency, intent_cats = intended_saliency(batch_size=hparams.batch_size, 
                                                             consistent=hparams.minibatch_consistency)
            
            # Rate prediction
            (rate_distribution, 
             pitch_distribution) = model_rate(feats.detach(), # .detach()
                                           mask_sample.detach(), 
                                           intent_saliency)
            
            loss_rate = criterion3(x, hparams, WSOLA, OLA, model_saliency, 
                                   rate_distribution, pitch_distribution, 
                                   mask_sample, intent_cats, criterion2, 
                                   uniform=True)
            
            
            reduced_loss_rate = loss_rate.item()
            loss_rate.backward()

            grad_norm_rate = torch.nn.utils.clip_grad_norm_(
                                                            model_rate.parameters(),
                                                            hparams.grad_clip_thresh,
                                                            )

            optimizer_rate.step()

            # Validation
            if (not math.isnan(reduced_loss_rate) and rank == 0):
                duration = time.perf_counter() - start
                print("Train loss {} {:.6f} Grad Norm Rate {:.6f} {:.2f}s/it".format(
                    iteration, reduced_loss_rate, grad_norm_rate, duration))
                logger.log_training_rate(reduced_loss_rate, grad_norm_rate, 
                                         learning_rate_rate, 
                                         hparams.exploitation_prob, 
                                         duration, iteration)

            if (iteration % hparams.iters_per_checkpoint == 0):
                validate(model_saliency, model_rate, WSOLA, OLA, criterion1, 
                         valset, collate_fn, iteration, hparams.batch_size, 
                         rate_classes, hparams.minibatch_consistency, n_gpus, 
                         logger, hparams.distributed_run, rank)
                
                if learning_rate_rate > hparams.learning_rate_lb:
                    learning_rate_rate *= hparams.learning_rate_decay
                
                if hparams.exploitation_prob < 0.85: #0.8
                    hparams.exploitation_prob *= hparams.exploration_decay
                
                # Saving the model
                if rank == 0:
                    checkpoint_path = os.path.join(output_directory, 
                                                   "checkpoint_{}".format(iteration))
                    save_checkpoint(model_rate, 
                                    optimizer_rate,
                                    learning_rate_rate,
                                    iteration, 
                                    checkpoint_path)

            iteration += 1
            # except Exception as ex:
            #     print(ex)

        sys.stdout.flush()


if __name__ == '__main__':
    hparams = create_hparams()

    hparams.output_directory = os.path.join(
                                        hparams.output_directory, 
                                        "VESUS_Block_Local_PitchRate_entropy_{}_exploit_{}_{}_max_3".format(
                                            hparams.lambda_entropy,
                                            hparams.exploitation_prob,
                                            hparams.extended_desc,
                                        )
                                    )

    if not hparams.output_directory:
        raise FileExistsError('Please specify the output dir.')
    else:
        if not os.path.exists(hparams.output_directory):
            os.mkdir(hparams.output_directory)

    # Record the hyper-parameters.
    hparams_snapshot_file = os.path.join(hparams.output_directory,
                                          'hparams.txt')
    with open(hparams_snapshot_file, 'w') as writer:
        pprint(hparams.__dict__, writer)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    train(
        hparams.output_directory, 
        hparams.log_directory,
        hparams.checkpoint_path_rate,
        hparams.checkpoint_path_saliency,
        hparams.warm_start,
        hparams.n_gpus,
        hparams.rank,
        hparams.group_name,
        hparams,
    )




































