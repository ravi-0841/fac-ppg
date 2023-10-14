#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 14:18:22 2023

@author: ravi
"""

import os
import time
import math
import torch
import numpy as np
from torch.utils.data import DataLoader
from on_the_fly_augmentor_raw_voice_mask import OnTheFlyAugmentor, acoustics_collate_raw
from block_pitch_duration_AC_decoupled import (MaskedRateModifier,
                                                RatePredictorActor,
                                                RatePredictorCritic)
from src.common.logger_PitchRatePred import SaliencyPredictorLogger
from src.common.hparams_actor_critic_vesus import create_hparams
from src.common.loss_function import EntropyLoss
from src.common.interpolation_block import (WSOLAInterpolation,
                                            BatchWSOLAInterpolation,
                                            BatchWSOLAInterpolationEnergy)
from src.common.pitch_modification_block import (PitchModification,
                                                 BatchPitchModification,
                                                 LocalPitchModification,
                                                 BatchLocalPitchModification)
from src.common.utils import (intended_saliency, 
                              get_random_mask_chunk, 
                              get_mask_blocks_inference)
from pprint import pprint

#%%
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
    model_actor = RatePredictorActor(temp_scale=1.).cuda()
    model_critic = RatePredictorCritic().cuda()
    return model_saliency, model_actor, model_critic


def warm_start_model(checkpoint_path, model_actor, model_critic):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_actor.load_state_dict(checkpoint_dict['state_dict_actor'])
    model_critic.load_state_dict(checkpoint_dict['state_dict_critic'])
    return model_actor, model_critic


def load_checkpoint_rate(checkpoint_path, model_actor, model_critic, 
                        optimizer_actor, optimizer_critic):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_actor.load_state_dict(checkpoint_dict['state_dict_actor'])
    model_critic.load_state_dict(checkpoint_dict['state_dict_critic'])
    optimizer_actor.load_state_dict(checkpoint_dict['optimizer_actor'])
    optimizer_critic.load_state_dict(checkpoint_dict['optimizer_critic'])
    learning_rate_actor = checkpoint_dict['learning_rate_actor']
    learning_rate_critic = checkpoint_dict['learning_rate_critic']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return (model_actor, model_critic, optimizer_actor,
            optimizer_critic, learning_rate_actor, 
            learning_rate_critic, iteration)


def load_checkpoint_saliency(checkpoint_path, model_saliency):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_saliency.load_state_dict(checkpoint_dict['state_dict'])
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint saliency '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model_saliency, iteration


def save_checkpoint(model_actor, model_critic, optimizer_actor, 
                    optimizer_critic, learning_rate_actor,
                    learning_rate_critic, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict_actor': model_actor.state_dict(),
                'state_dict_critic': model_critic.state_dict(),
                'optimizer_actor': optimizer_actor.state_dict(),
                'optimizer_critic': optimizer_critic.state_dict(),
                'learning_rate_actor': learning_rate_actor,
                'learning_rate_critic': learning_rate_critic}, filepath)
    
#%% Validation
def validate(model_saliency, model_actor, WSOLA, OLA, valset, 
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
        for i, batch in enumerate(val_loader):
            x, em = batch[0].to("cuda"), batch[1].to("cuda")
            intent, cats = intended_saliency(batch_size=batch_size, 
                                             consistent=consistency,
                                             relative_prob=[0., 0.34, 0.33, 0.33, 0.])
            feats, posterior, mask_sample, orig_pred = model_saliency(x, em)
            mask_sample = get_random_mask_chunk(mask_sample)

            (rate_distribution,
             pitch_distribution) = model_actor(feats, mask_sample, intent)
            index_rate = torch.argmax(rate_distribution, dim=-1)
            index_pitch = torch.argmax(pitch_distribution, dim=-1)
            
            # rate = 0.5 + 0.1*index_rate # 0.2*index
            # pitch = 0.5 + 0.1*index_pitch # 0.2*index
            
            rate = 0.25 + 0.15*index_rate # 0.2*index
            pitch = 0.25 + 0.15*index_pitch # 0.2*index
            
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
            
        val_loss = val_loss / (i + 1)

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

#%%
def environment(hparams, x, WSOLA, OLA, 
                duration_factor, pitch_factor, 
                mask):
    pitch_mod_speech = OLA(mask=mask[:,:,0], factor=pitch_factor, speech=x)
    mod_speech, mod_energy, _ = WSOLA(mask=mask[:,:,0], 
                                      rate=duration_factor, 
                                      speech=pitch_mod_speech)
    mod_speech = mod_speech.to("cuda")
    mod_energy = mod_energy.to("cuda")
    
    return mod_speech, mod_energy

#%%

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
    
    entropy_criterion = EntropyLoss()

    model_saliency.eval()
    model_rate.train()

    # ================ MAIN TRAINING LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            try:
                start = time.perf_counter()
                for param_group in optimizer_rate.param_groups:
                    param_group['lr'] = learning_rate_rate
                
                # Intended Saliency
                intent_saliency, intent_cats = intended_saliency(batch_size=hparams.batch_size, 
                                                                 consistent=hparams.minibatch_consistency,
                                                                 relative_prob=[0., 0.3, 0.3, 0.3, 0.1])

                # model_rate.zero_grad()

                (x, e, l) = (batch[0].to("cuda"), batch[1].to("cuda"),
                              batch[3])
                l = torch.div(l, hparams.downsampling_factor, 
                              rounding_mode="floor")

                # input_shape should be [#batch_size, 1, #time]
                feats, posterior, mask_sample, y_pred = model_saliency(x, e)
                # chunked_masks, chunks = get_mask_blocks_inference(mask_sample)
                mask_sample = get_random_mask_chunk(mask_sample)

                value, rate_dist, pitch_dist = model_rate(feats.detach(), 
                                                          mask_sample.detach(), 
                                                          intent_saliency)
                
                index_rate = torch.multinomial(rate_dist, 1, 
                                          replacement=True) #explore
                index_pitch = torch.multinomial(pitch_dist, 1, 
                                          replacement=True) #explore
                
                rate = 0.25 + 0.15*index_rate
                pitch = 0.25 + 0.15*index_pitch
                
                # Computing Q-value after taking action
                updated_signal, updated_energy = environment(hparams, x, WSOLA, OLA, 
                                                             rate, pitch, mask_sample)
                _, _, _, updated_saliency = model_saliency(updated_signal, updated_energy)
                Q_value = updated_saliency.gather(1, intent_cats.view(-1,1)).view(-1)
                
                # Computing advantage
                advantage = Q_value.detach() - value.view(-1)
                
                # Actor loss term
                actor_loss_rate = torch.mean(torch.mul(-torch.log(rate_dist.gather(1, index_rate.view(-1,1)).view(-1)), 
                                              advantage))
                actor_loss_pitch = torch.mean(torch.mul(-torch.log(pitch_dist.gather(1, index_pitch.view(-1,1)).view(-1)), 
                                              advantage))
                actor_loss = actor_loss_rate + actor_loss_pitch
                
                # Critic loss term
                critic_loss = torch.mean(torch.abs(advantage))
                
                # Entropy loss term
                entropy_loss = -entropy_criterion(pitch_dist) -entropy_criterion(rate_dist)

                # Combining all three losses            
                actor_critic_loss = actor_loss + hparams.lambda_critic*critic_loss + hparams.lambda_entropy*entropy_loss
                
                # Updating the parameters
                optimizer_rate.zero_grad()
                actor_critic_loss.backward()
                grad_norm_rate = torch.nn.utils.clip_grad_norm_(
                                                                model_rate.parameters(),
                                                                hparams.grad_clip_thresh,
                                                                )
                optimizer_rate.step()

                # Validation
                if (not math.isnan(actor_loss.item()) and not math.isnan(critic_loss.item()) and rank == 0):
                    duration = time.perf_counter() - start
                    print("Train loss {} Actor: {:.6f}, Critic: {:.6f} Grad Norm Rate {:.6f} {:.2f}s/it".format(
                        iteration, actor_loss.item(), critic_loss.item(), grad_norm_rate, duration))
                    print("Predicted Value: {:.4f} Q_value: {:.4f}".format(
                                                                            torch.mean(value).item(), 
                                                                            torch.mean(Q_value).item()))
                    logger.log_training_rate(actor_critic_loss.item(), 
                                             grad_norm_rate, 
                                             learning_rate_rate, 
                                             hparams.exploitation_prob, 
                                             duration, iteration)

                if (iteration % hparams.iters_per_checkpoint == 0):
                    validate(model_saliency, model_rate, WSOLA, OLA, valset, 
                             collate_fn, iteration, hparams.batch_size, 
                             rate_classes, hparams.minibatch_consistency, n_gpus, 
                             logger, hparams.distributed_run, rank)
                    
                    if learning_rate_rate > hparams.learning_rate_lb:
                        learning_rate_rate *= hparams.learning_rate_decay
                    
                #     if hparams.exploitation_prob < 0.85: #0.8
                #         hparams.exploitation_prob *= hparams.exploration_decay
                    
                #     # Saving the model
                    if rank == 0:
                        checkpoint_path = os.path.join(output_directory, 
                                                        "checkpoint_{}".format(iteration))
                        save_checkpoint(model_rate, 
                                        optimizer_rate,
                                        learning_rate_rate,
                                        iteration, 
                                        checkpoint_path)

                iteration += 1
            except Exception as ex:
                print(ex)

#%% Main function
if __name__ == '__main__':
    hparams = create_hparams()

    hparams.output_directory = os.path.join(
                                        hparams.output_directory, 
                                        "VESUS_Block_entropy_{}_actor_critic_{}_annealedLR".format(
                                            hparams.lambda_entropy,
                                            hparams.lambda_critic,
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
