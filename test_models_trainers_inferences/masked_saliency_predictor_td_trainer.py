#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  28 15:30:44 2023

@author: ravi
"""


import os
import time
import math
import torch
import numpy as np
from torch.utils.data import DataLoader
from saliency_predictor_time_domain import SaliencyPredictor
from on_the_fly_augmentor_raw import OnTheFlyAugmentor, acoustics_collate_raw
from src.common.loss_function import (MaskedSpectrogramL1LossReduced,
                                        ExpectedKLDivergence,
                                        VecExpectedKLDivergence, 
                                        SparsityKLDivergence,
                                    )
from src.common.logger_SaliencyPred_timeDomain import SaliencyPredictorLogger
from src.common.hparams_onflyaugmentor import create_hparams
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
    model = SaliencyPredictor(hparams.temp_scale).cuda()
    return model


def warm_start_model(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def validate(model, criterion, valset, collate_fn, iteration, 
             batch_size, n_gpus, logger, distributed_run, rank):
    """Handles all the validation scoring and printing"""
    model.eval()
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
            posterior, mask_sample, y_pred = model(x)
            loss = criterion(y_pred, y)
            reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)

    model.train()
    if rank == 0:
        print("Validation loss {}: {:9f}  ".format(iteration, val_loss))
        logger.log_validation(
                                val_loss,
                                model,
                                x,
                                y,
                                y_pred,
                                posterior[:,:,1].squeeze(),
                                # torch.argmax(mask_sample,-1).squeeze(),
                                mask_sample[:,:,0:1].squeeze(),
                                iteration,
                            )


def train(output_directory, log_directory, checkpoint_path, warm_start, n_gpus,
          rank, group_name, hparams):
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

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    criterion1 = VecExpectedKLDivergence(alpha=hparams.alpha, 
                                        beta=hparams.beta)
    criterion2 = torch.nn.L1Loss() #MSELoss
    criterion3 = SparsityKLDivergence()

    logger = prepare_directories_and_logger(
        output_directory, log_directory, rank)

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path:
        if warm_start:
            model = warm_start_model(checkpoint_path, model)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))
            
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of trainable parameters are: ", num_params)

    model.train()
    # ================ MAIN TRAINING LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            model.zero_grad()
            x, y, l = batch[0].to("cuda"), batch[1].to("cuda"), batch[2]
            l = torch.div(l, 160, rounding_mode="floor")
            # input_shape should be [#batch_size, 1, #time]

            posterior, mask_sample, y_pred = model(x, pre_computed_mask=None)

            loss = (
                    hparams.lambda_prior_KL*criterion1(posterior.squeeze(), l)
                    + hparams.lambda_predict*criterion2(y_pred, y)
                    + hparams.lambda_sparse_KL*criterion3(posterior)
                )
            reduced_loss = loss.item()

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                                                        model.parameters(),
                                                        hparams.grad_clip_thresh,
                                                    )

            optimizer.step()

            if not math.isnan(reduced_loss) and rank == 0:
                duration = time.perf_counter() - start
                print("Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                    iteration, reduced_loss, grad_norm, duration))
                logger.log_training(
                    reduced_loss, grad_norm, learning_rate, duration, iteration)

            if (iteration % hparams.iters_per_checkpoint == 0):
                validate(model, criterion2, valset, collate_fn, 
                         iteration, hparams.batch_size, n_gpus, logger, 
                         hparams.distributed_run, rank)
                if learning_rate > hparams.learning_rate_lb:
                    learning_rate *= hparams.learning_rate_decay
                
                # Saving the model
                if rank == 0:
                    checkpoint_path = os.path.join(
                        output_directory, "checkpoint_{}".format(iteration))
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path)

            iteration += 1


if __name__ == '__main__':
    hparams = create_hparams()

    hparams.output_directory = os.path.join(
                                        hparams.output_directory, 
                                        "timeDomain_{}_{}_{}_{}_{}".format(
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







































