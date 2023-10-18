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
from src.common.logger_ActorCritic import SaliencyPredictorLogger
from src.common.hparams_actor_critic_decoupled_vesus import create_hparams
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


def load_checkpoint_AC(checkpoint_path, model_actor, model_critic, 
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
    model_actor.eval()
    with torch.no_grad():
        val_loader = DataLoader(valset,
                                sampler=None,
                                num_workers=4,
                                shuffle=True,
                                batch_size=batch_size,
                                collate_fn=collate_fn,
                                drop_last=True)

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, em = batch[0].to("cuda"), batch[1].to("cuda")
            intent, cats = intended_saliency(batch_size=batch_size, 
                                             consistent=consistency,
                                             relative_prob=[0., 0.25, 0.25, 0.25, 0.25])
            feats, posterior, mask_sample, orig_pred = model_saliency(x, em)
            mask_sample = get_random_mask_chunk(mask_sample)

            (rate_distribution,
             pitch_distribution) = model_actor(x, mask_sample, intent)
            index_rate = torch.argmax(rate_distribution, dim=-1)
            index_pitch = torch.argmax(pitch_distribution, dim=-1)
            
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

    model_actor.train()
    if rank == 0:
        print("Validation loss {}: {:9f}  ".format(iteration, val_loss))
        logger.log_validation(
                                val_loss,
                                model_saliency,
                                model_actor,
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
def create_modified_patches(active_chunks, sampled_chunk, rate, total):
    z = np.empty((0,))
    # t = [(20, 49, 30), (60, 80, 21)]
    # total = 200
    proxy_end = 0
    final_zeros = total - active_chunks[-1][1] - 1
    for i, chunk in enumerate(active_chunks):
        z = np.concatenate((z, np.zeros((chunk[0] - proxy_end,))), axis=0)
        if sampled_chunk == i:
            z = np.concatenate((z, np.ones((int(np.ceil(chunk[2]*rate)),))), axis=0)
        else:
            z = np.concatenate((z, np.ones((chunk[2],))), axis=0)
        proxy_end = chunk[1]+1
    z = np.concatenate((z, np.zeros((final_zeros,))), axis=0)
    return z

#%%
def updated_environment(hparams, x, WSOLA, OLA, 
                        duration_factor, pitch_factor, 
                        mask, active_chunks, sampled_chunks):
    # mask -> [batch, #Time, 512]
    # x -> [batch, 1, #time]
    # active_chunks -> list of size #batch, each element is active regions
    # sampled_chunks -> list of size #batch, each element is sampled region

    pitch_mod_speech = OLA(mask=mask[:,:,0], factor=pitch_factor, speech=x)
    mod_speech, mod_energy, _ = WSOLA(mask=mask[:,:,0], 
                                      rate=duration_factor, 
                                      speech=pitch_mod_speech)
    mod_speech = mod_speech.to("cuda")
    mod_energy = mod_energy.to("cuda")
    
    # Carrying out mask transformation
    mask = mask.detach().cpu().numpy()[:,:,0]
    updated_masks = []
    max_len = 0
    for i in range(mask.shape[0]):
        updated_mask = create_modified_patches(active_chunks[i], 
                                               sampled_chunks[i], 
                                               duration_factor[i].item(), 
                                               len(mask[i]))
        updated_masks.append(updated_mask)
        max_len = max(max_len, len(updated_mask))

    mod_mask = [np.concatenate((u, np.zeros((max_len - len(u),))), axis=0) for u in updated_masks]
    mod_mask = torch.from_numpy(np.asarray(mod_mask)).float().to("cuda")
    mod_mask = mod_mask[:,:mod_energy.size()[2]].unsqueeze(dim=-1).repeat(1,1,512)
    return mod_speech, mod_energy, mod_mask

#%%

def train(output_directory, log_directory, checkpoint_path_AC,
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

    model_saliency, model_actor, model_critic = load_model(hparams)
    learning_rate_actor = hparams.learning_rate_actor
    learning_rate_critic = hparams.learning_rate_critic

    optimizer_actor = torch.optim.Adam(model_actor.parameters(), 
                                      lr=learning_rate_actor, 
                                      weight_decay=hparams.weight_decay)
    optimizer_critic = torch.optim.Adam(model_critic.parameters(),
                                        lr=learning_rate_critic,
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
    
    if checkpoint_path_AC:
        if warm_start:
            model_actor, model_critic = warm_start_model(checkpoint_path_AC,
                                                        model_actor, model_critic)
        else:
            (
                model_actor,
                model_critic,
                optimizer_actor,
                optimizer_critic,
                _learning_rate_actor,
                _learning_rate_critic,
                iteration,
            ) = load_checkpoint_AC(checkpoint_path_AC,
                                    model_actor,
                                    model_critic,
                                    optimizer_actor,
                                    optimizer_critic)
            if hparams.use_saved_learning_rate:
                learning_rate_actor = _learning_rate_actor
                learning_rate_critic = _learning_rate_critic

            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    num_params = sum(p.numel() for p in model_actor.parameters() if p.requires_grad)
    num_params += sum(p.numel() for p in model_critic.parameters() if p.requires_grad)
    print("Total number of trainable parameters are: ", num_params)
    
    WSOLA = BatchWSOLAInterpolationEnergy(win_size=hparams.win_length, 
                                   hop_size=hparams.hop_length,
                                   tolerance=hparams.hop_length,
                                   thresh=1e-3)
    OLA = BatchLocalPitchModification(frame_period=10)
    
    entropy_criterion = EntropyLoss()

    model_saliency.eval()
    model_actor.train()
    model_critic.train()

    # ================ MAIN TRAINING LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            try:
                start = time.perf_counter()
                for param_group in optimizer_actor.param_groups:
                    param_group['lr'] = learning_rate_actor
                for param_group in optimizer_critic.param_groups:
                    param_group['lr'] = learning_rate_critic
                
                # Intended Saliency
                (intent_saliency, 
                intent_cats) = intended_saliency(batch_size=hparams.batch_size, 
                                                consistent=hparams.minibatch_consistency,
                                                relative_prob=[0., 0.25, 0.25, 0.25, 0.25])

                (x, e, l) = (batch[0].to("cuda"), batch[1].to("cuda"),
                              batch[3])
                l = torch.div(l, hparams.downsampling_factor, 
                              rounding_mode="floor")

                # input_shape should be [#batch_size, 1, #time]
                feats, posterior, mask_sample, y_pred = model_saliency(x, e)
                mask_sample, active_chunks, sampled_chunks = get_random_mask_chunk(mask_sample, 
                                                                                   sampling_info=True)

                # Get the action distribution
                rate_dist, pitch_dist = model_actor(x, mask_sample.detach(),
                                                    intent_saliency)
                # Getting critic values
                value = model_critic(x, mask_sample.detach(), intent_saliency)
                
                # Sample an action
                index_rate = torch.multinomial(rate_dist, 1, 
                                          replacement=True) #explore
                index_pitch = torch.multinomial(pitch_dist, 1, 
                                          replacement=True) #explore
                
                rate = 0.25 + 0.15*index_rate
                pitch = 0.25 + 0.15*index_pitch

                # Take the action and update the state
                (updated_signal, 
                 updated_energy, 
                 updated_mask) = updated_environment(hparams, x, WSOLA, OLA, 
                                                     rate, pitch, mask_sample,
                                                     active_chunks, sampled_chunks)
                
                # Computing Q-value after taking action
                _, _, _, updated_saliency = model_saliency(x=updated_signal, e=updated_energy, 
                                                           pre_computed_mask=updated_mask)
                Q_value = updated_saliency.gather(1, intent_cats.view(-1,1)).view(-1)
                
                # Computing advantage
                advantage = Q_value.detach() - value.view(-1)
                
                # Actor loss term
                actor_loss_rate = torch.mean(torch.mul(-torch.log(rate_dist.gather(1, index_rate.view(-1,1)).view(-1)), 
                                              advantage.detach()))
                actor_loss_pitch = torch.mean(torch.mul(-torch.log(pitch_dist.gather(1, index_pitch.view(-1,1)).view(-1)), 
                                              advantage.detach()))
                actor_loss = actor_loss_rate + actor_loss_pitch
                
                # Critic loss term
                critic_loss = torch.mean(torch.abs(advantage))
                
                # Entropy loss term
                entropy_loss = -entropy_criterion(pitch_dist) -entropy_criterion(rate_dist)

                # Combining all three losses            
                actor_total_loss = actor_loss + hparams.lambda_entropy*entropy_loss
                
                # Updating the Critic parameters
                optimizer_critic.zero_grad()
                critic_loss.backward()
                grad_norm_critic = torch.nn.utils.clip_grad_norm_(
                                                                model_critic.parameters(),
                                                                hparams.grad_clip_thresh,
                                                                )
                optimizer_critic.step()
                
                # Updating the Actor parameters
                optimizer_actor.zero_grad()
                actor_total_loss.backward()
                grad_norm_actor = torch.nn.utils.clip_grad_norm_(
                                                                model_actor.parameters(),
                                                                hparams.grad_clip_thresh,
                                                                )
                optimizer_actor.step()

                # Validation
                if (not math.isnan(actor_loss.item()) and not math.isnan(critic_loss.item()) and rank == 0):
                    duration = time.perf_counter() - start
                    print("Train loss {} Actor: {:.6f}, Critic: {:.6f}, Grad Norm Actor {:.6f}, Grad Norm Critic {:.6f}, {:.2f}s/it".format(
                        iteration, actor_loss.item(), critic_loss.item(), grad_norm_actor, grad_norm_critic, duration))
                    print("Predicted Value: {:.4f} Q_value: {:.4f}".format(torch.mean(value).item(), 
                                                                           torch.mean(Q_value).item()))
                    logger.log_training_rate(actor_total_loss.item(),
                                             critic_loss.item(),
                                             grad_norm_actor, 
                                             grad_norm_critic,
                                             learning_rate_actor,
                                             learning_rate_critic,
                                             hparams.exploitation_prob, 
                                             duration, iteration)

                if (iteration % hparams.iters_per_checkpoint == 0):
                    validate(model_saliency, model_actor, WSOLA, OLA, valset, 
                              collate_fn, iteration, hparams.batch_size, 
                              rate_classes, hparams.minibatch_consistency, n_gpus, 
                              logger, hparams.distributed_run, rank)
                    
                    # if learning_rate_rate > hparams.learning_rate_lb:
                    #     learning_rate_rate *= hparams.learning_rate_decay
                    
                    # if hparams.exploitation_prob < 0.85: #0.8
                    #     hparams.exploitation_prob *= hparams.exploration_decay
                    
                    # Saving the model
                    if rank == 0:
                        checkpoint_path = os.path.join(output_directory, 
                                                        "checkpoint_{}".format(iteration))
                        save_checkpoint(model_actor,
                                        model_critic,
                                        optimizer_actor,
                                        optimizer_critic,
                                        learning_rate_actor,
                                        learning_rate_critic,
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
                                        "VESUS_Block_entropy_{}_actor_critic_{}_decoupled".format(
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
