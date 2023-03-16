#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 17:55:47 2023

@author: ravi
"""

import os
import time
import math
import torch
import numpy as np
from torch.utils.data import DataLoader
# from saliency_predictor_TD_RL_postEmoRate import MaskedRateModifier, RatePredictor
from lstm_gen_conv_mask_trans_masked_postEmoRate import MaskedRateModifier, RatePredictor
from on_the_fly_augmentor_raw import OnTheFlyAugmentor, acoustics_collate_raw
from src.common.loss_function import (MaskedSpectrogramL1LossReduced,
                                        ExpectedKLDivergence,
                                        VecExpectedKLDivergence, 
                                        SparsityKLDivergence,
                                    )
from src.common.logger_SaliencyPred_timeDomain import SaliencyPredictorLogger
from src.common.hparams_onflyaugmentor import create_hparams
from src.common.interpolation_block import WSOLAInterpolation, BatchWSOLAInterpolation
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
    model_rate = RatePredictor(temp_scale=1.0).cuda()
    return model_saliency, model_rate


def warm_start_model(checkpoint_path, model_saliency, model_rate):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_saliency.load_state_dict(checkpoint_dict['state_dict_saliency'])
    model_rate.load_state_dict(checkpoint_dict['state_dict_rate'])
    return model_saliency, model_rate


def load_checkpoint(checkpoint_path, model_saliency, 
                    model_rate, optimizer_saliency, 
                    optimizer_rate):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_saliency.load_state_dict(checkpoint_dict['state_dict_saliency'])
    optimizer_saliency.load_state_dict(checkpoint_dict['optimizer_saliency'])
    model_rate.load_state_dict(checkpoint_dict['state_dict_rate'])
    optimizer_rate.load_state_dict(checkpoint_dict['optimizer_rate'])
    learning_rate_saliency = checkpoint_dict['learning_rate_saliency']
    learning_rate_rate = checkpoint_dict['learning_rate_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return (model_saliency, model_rate, optimizer_saliency, 
            optimizer_rate, learning_rate_saliency, 
            learning_rate_rate, iteration)


def save_checkpoint(model_saliency, model_rate, optimizer_saliency, 
                    optimizer_rate, learning_rate_saliency, 
                    learning_rate_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict_saliency': model_saliency.state_dict(),
                'state_dict_rate': model_rate.state_dict(),
                'optimizer_saliency': optimizer_saliency.state_dict(),
                'optimizer_rate': optimizer_rate.state_dict(),
                'learning_rate_saliency': learning_rate_saliency,
                'learning_rate_rate': learning_rate_rate}, filepath)


def validate(model_saliency, model_rate, criterion, valset, 
             collate_fn, iteration, batch_size, rate_classes, 
             n_gpus, logger, distributed_run, rank):
    """Handles all the validation scoring and printing"""
    model_saliency.eval()
    model_rate.eval()
    with torch.no_grad():
        val_loader = DataLoader(
                                valset,
                                sampler=None,
                                num_workers=4,
                                shuffle=True,
                                batch_size=batch_size,
                                collate_fn=collate_fn,
                            )

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y, _ = batch[0].to("cuda"), batch[1].to("cuda"), batch[2]
            e = intended_saliency(batch_size)
            feats, posterior, mask_sample, y_pred = model_saliency(x)
            loss = criterion(y_pred, y)
            reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
            rate_distribution = model_rate(feats, posterior, e)
        val_loss = val_loss / (i + 1)

    model_saliency.train()
    model_rate.train()
    if rank == 0:
        print("Validation loss {}: {:9f}  ".format(iteration, val_loss))
        logger.log_validation(
                                val_loss,
                                model_saliency,
                                model_rate,
                                x,
                                y,
                                y_pred,
                                posterior[:,:,1:2],
                                mask_sample[:,:,0:1],
                                rate_distribution,
                                rate_classes,
                                iteration,
                            )
        # logger_rate.log_parameters(model_rate, iteration)


def intended_saliency(batch_size, relative_prob=[0.0, 0.25, 0.25, 0.25, 0.25]):
    emotion_cats = torch.multinomial(torch.Tensor(relative_prob), 
                                      batch_size,
                                      replacement=True)
    # emotion_cats = torch.multinomial(torch.Tensor(relative_prob), 1).repeat(batch_size)
    emotion_codes = torch.nn.functional.one_hot(emotion_cats, 5).float().to("cuda")

    # index_intent = torch.multinomial(torch.Tensor(relative_prob), 1)
    # intent_saliency = torch.zeros(hparams.batch_size, 5)
    # intent_saliency[:, index_intent[0]] = 1.0
    # intent_saliency = intent_saliency.to("cuda")
    return emotion_codes


def train(output_directory, log_directory, checkpoint_path, 
          warm_start, n_gpus, rank, group_name, hparams):
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
    learning_rate_saliency = hparams.learning_rate_saliency
    learning_rate_rate = hparams.learning_rate_rate
    optimizer_saliency = torch.optim.Adam(model_saliency.parameters(), 
                                          lr=learning_rate_saliency, 
                                          weight_decay=hparams.weight_decay)
    optimizer_rate = torch.optim.Adam(model_rate.parameters(), 
                                          lr=learning_rate_rate, 
                                          weight_decay=hparams.weight_decay)

    criterion1 = VecExpectedKLDivergence(alpha=hparams.alpha, 
                                        beta=hparams.beta)
    criterion2 = torch.nn.L1Loss() #MSELoss
    criterion3 = SparsityKLDivergence()

    logger = prepare_directories_and_logger(output_directory, log_directory, rank)

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)
    
    rate_classes = [str(np.round(x,2)) for x in np.arange(0.5, 1.6, 0.2)]
    # rate_classes = [str(np.round(x,2)) for x in np.arange(0.5, 1.6, 0.1)]

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path:
        if warm_start:
            model_saliency, model_rate = warm_start_model(checkpoint_path, 
                                                          model_saliency,
                                                          model_rate)
        else:
            (
                model_saliency,
                model_rate,
                optimizer_saliency,
                optimizer_rate,
                _learning_rate_saliency,
                _learning_rate_rate,
                iteration,
            ) = load_checkpoint(
                                checkpoint_path,
                                model_saliency,
                                model_rate,
                                optimizer_saliency,
                                optimizer_rate,
                                )
            if hparams.use_saved_learning_rate:
                learning_rate_saliency = _learning_rate_saliency
                learning_rate_rate = _learning_rate_rate

            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))
            
    num_params = sum(p.numel() for p in model_saliency.parameters() if p.requires_grad)
    num_params += sum(p.numel() for p in model_rate.parameters() if p.requires_grad)
    print("Total number of trainable parameters are: ", num_params)
    
    WSOLA = BatchWSOLAInterpolation(win_size=hparams.win_length, 
                                   hop_size=hparams.hop_length,
                                   tolerance=hparams.hop_length)

    model_saliency.train()
    model_rate.train()
    # ================ MAIN TRAINING LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            try:
                start = time.perf_counter()
                for param_group in optimizer_saliency.param_groups:
                    param_group['lr'] = learning_rate_saliency
                for param_group in optimizer_rate.param_groups:
                    param_group['lr'] = learning_rate_rate
    
                model_saliency.zero_grad()
                model_rate.zero_grad()
    
                x, y, l = batch[0].to("cuda"), batch[1].to("cuda"), batch[2]
                l = torch.div(l, hparams.downsampling_factor, 
                              rounding_mode="floor")
                # input_shape should be [#batch_size, 1, #time]
    
                feats, posterior, mask_sample, y_pred = model_saliency(x)
    
                loss_saliency = (
                                    hparams.lambda_prior_KL*criterion1(posterior, l)
                                    + hparams.lambda_predict*criterion2(y_pred, y)
                                    + hparams.lambda_sparse_KL*criterion3(posterior)
                                )
                reduced_loss_saliency = loss_saliency.item()
                
                # Intended Saliency
                intent_saliency = intended_saliency(hparams.batch_size)
                
                # Rate prediction
                rate_distribution = model_rate(feats.detach(),
                                               posterior.detach(),
                                               intent_saliency)
                index = torch.multinomial(rate_distribution, 1)
                rate = 0.5 + 0.2*index
                mod_speech, _ = WSOLA(mask=mask_sample[:,:,0], 
                                         rate=rate, speech=x)
            
                mod_speech = mod_speech.to("cuda")
                with torch.no_grad():
                    _, _, _, s = model_saliency(mod_speech)

                loss_rate = torch.mean(torch.abs(s - intent_saliency), dim=-1)
                loss_rate = torch.mean(loss_rate.detach() * rate_distribution.gather(1, index.view(-1,1)))
                reduced_loss_rate = loss_rate.item()
                
                total_loss = loss_rate + loss_saliency
                total_loss.backward()
                grad_norm_saliency = torch.nn.utils.clip_grad_norm_(
                                                                    model_saliency.parameters(),
                                                                    hparams.grad_clip_thresh,
                                                                    )

                grad_norm_rate = torch.nn.utils.clip_grad_norm_(
                                                                model_rate.parameters(),
                                                                hparams.grad_clip_thresh,
                                                                )

                optimizer_saliency.step()
                optimizer_rate.step()
    
                # Validation
                if (not math.isnan(reduced_loss_saliency) 
                    and not math.isnan(reduced_loss_rate) 
                    and rank == 0):
                    duration = time.perf_counter() - start
                    print("Train loss {} {:.6f} Grad Norm Saliency {:.6f} {:.2f}s/it".format(
                        iteration, reduced_loss_saliency, grad_norm_saliency, duration))
                    print("Train loss {} {:.6f} Grad Norm Rate {:.6f} {:.2f}s/it".format(
                        iteration, reduced_loss_rate, grad_norm_rate, duration))
                    logger.log_training_saliency(reduced_loss_saliency, grad_norm_saliency, 
                                                 learning_rate_saliency, duration, 
                                                 iteration)
                    logger.log_training_rate(reduced_loss_rate, grad_norm_rate, 
                                             learning_rate_rate, duration, 
                                             iteration)

                if (iteration % hparams.iters_per_checkpoint == 0):
                    validate(model_saliency, model_rate, criterion2, valset, 
                             collate_fn, iteration, hparams.batch_size, 
                             rate_classes, n_gpus, logger, 
                             hparams.distributed_run, rank)
                    if learning_rate_saliency > hparams.learning_rate_lb:
                        learning_rate_saliency *= hparams.learning_rate_decay
                    # if learning_rate_rate > hparams.learning_rate_lb:
                    learning_rate_rate *= (1/hparams.learning_rate_decay)
                    
                    # Saving the model
                    if rank == 0:
                        checkpoint_path = os.path.join(output_directory, 
                                                       "checkpoint_{}".format(iteration))
                        save_checkpoint(model_saliency, 
                                        model_rate, 
                                        optimizer_saliency,
                                        optimizer_rate,
                                        learning_rate_saliency,
                                        learning_rate_rate,
                                        iteration, 
                                        checkpoint_path)

                iteration += 1
            except Exception as ex:
                print(ex)


if __name__ == '__main__':
    hparams = create_hparams()

    hparams.output_directory = os.path.join(
                                        hparams.output_directory, 
                                        "lr_opposing_{}_{}_{}_{}_{}".format(
                                            hparams.lambda_prior_KL,
                                            hparams.lambda_predict,
                                            hparams.lambda_sparse_KL,
                                            hparams.temp_scale,
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
        hparams.checkpoint_path,
        hparams.warm_start,
        hparams.n_gpus,
        hparams.rank,
        hparams.group_name,
        hparams,
    )
