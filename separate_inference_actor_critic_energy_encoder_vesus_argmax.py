#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 14:19:09 2023

@author: ravi
"""


import os
import sys
import time
import math
import torch
import pylab
import numpy as np
import seaborn as sns
import scipy.stats as scistat
import soundfile as sf
import joblib

from torch.utils.data import DataLoader
from block_pitch_duration_energy_AC_encoder import MaskedRateModifier, RatePredictorAC
from on_the_fly_augmentor_raw_voice_mask import OnTheFlyAugmentor, acoustics_collate_raw
from src.common.utils import (median_mask_filtering, 
                              refining_mask_sample,
                              get_mask_blocks_inference,
                              )
from src.common.hparams_actor_critic_energy_vesus import create_hparams
from src.common.interpolation_block import WSOLAInterpolationBlockEnergy
from src.common.pitch_energy_modification_block import LocalPitchEnergyModification
from pprint import pprint


#%%
def prepare_dataloaders(hparams, valid=True):
    # Get data, data loaders and collate function ready
    if valid:
        testset = OnTheFlyAugmentor(
                            utterance_paths_file=hparams.validation_files,
                            hparams=hparams,
                            augment=False,
                        )
    else:
        testset = OnTheFlyAugmentor(
                            utterance_paths_file=hparams.testing_files,
                            hparams=hparams,
                            augment=False,
                        )

    hparams.load_feats_from_disk = False
    hparams.is_cache_feats = False
    hparams.feats_cache_path = ''

    collate_fn = acoustics_collate_raw
    
    test_loader = DataLoader(
                            testset,
                            num_workers=1,
                            shuffle=True,
                            sampler=None,
                            batch_size=1,
                            drop_last=False,
                            collate_fn=collate_fn,
                            )
    return test_loader, collate_fn


def load_model(hparams):
    model_saliency = MaskedRateModifier(hparams.temp_scale).cuda()
    model_rate = RatePredictorAC(temp_scale=1.).cuda()
    return model_saliency, model_rate


def load_checkpoint_rate(checkpoint_path, model_rate):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_rate.load_state_dict(checkpoint_dict['state_dict_rate'])
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model_rate, iteration


def load_checkpoint_saliency(checkpoint_path, model_saliency):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_saliency.load_state_dict(checkpoint_dict['state_dict'])
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint saliency '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model_saliency, iteration


def intended_saliency(batch_size, relative_prob=[0.0, 1.0, 0.0, 0.0, 0.0]):
    emotion_cats = torch.multinomial(torch.Tensor(relative_prob), 
                                     batch_size, replacement=True)
    emotion_codes = torch.nn.functional.one_hot(emotion_cats, 5).float().to("cuda")
    return emotion_codes


def plot_figures(feats, waveform, mod_waveform, posterior, 
                 mask, y, y_pred, rate_dist, pitch_dist, 
                 energy_dist, iteration, hparams):
    
    mask_thresh = np.zeros((len(mask), ))
    mask_thresh[np.where(mask>0)[0]] = 1

    # Plotting details
    pylab.xticks(fontsize=18)
    pylab.yticks(fontsize=18)
    fig, ax = pylab.subplots(7, 1, figsize=(40, 24))
    
    ax[0].plot(waveform, linewidth=1.5, label="original")
    ax[0].plot(mod_waveform, linewidth=1.5, label="modified")
    ax[0].legend()
    ax[0].set_xlabel('Time',fontsize=15) #xlabel
    ax[0].set_ylabel('Magnitude', fontsize=15) #ylabel
    # pylab.tight_layout()

    ax[1].plot(posterior, linewidth=2.5, color='g')
    ax[1].plot(mask_thresh, "b", linewidth=2.5)
    ax[1].set_xlabel('Frame',fontsize=15) #xlabel
    ax[1].set_ylabel('Probability', fontsize=15) #ylabel
    # pylab.tight_layout()
    
    classes = ["neu", "ang", "hap", "sad", "fea"]
    ax[2].bar(classes, y, alpha=0.5, label="target")
    ax[2].bar(classes, y_pred, alpha=0.5, label="pred")
    ax[2].legend(loc=1)
    ax[2].set_xlabel('Classes',fontsize=15) #xlabel
    ax[2].set_ylabel('Softmax Score', fontsize=15) #ylabel
    # pylab.tight_layout()
    
    ax[3].imshow(feats, aspect="auto", origin="lower",
                   interpolation='none')
    ax[3].plot(257*mask_thresh, "w", linewidth=4.0)
    ax[3].set_xlabel('Frame',fontsize=15) #xlabel
    ax[3].set_ylabel('Dimension', fontsize=15) #ylabel
    # pylab.tight_layout()
    
    classes = [str(np.round(r,1)) for r in np.arange(0.5, 1.6, 0.1)]
    ax[4].bar(classes, rate_dist, alpha=0.5, color="r", label="Duration")
    ax[4].legend(loc=1)
    ax[4].set_xlabel('Classes',fontsize=15) #xlabel
    ax[4].set_ylabel('Softmax Score', fontsize=15) #ylabel
    # pylab.tight_layout()
    
    classes = [str(np.round(r,1)) for r in np.arange(0.5, 1.6, 0.1)]
    ax[5].bar(classes, pitch_dist, alpha=0.5, color="g", label="Pitch")
    ax[5].legend(loc=1)
    ax[5].set_xlabel('Classes',fontsize=15) #xlabel
    ax[5].set_ylabel('Softmax Score', fontsize=15) #ylabel
    # pylab.tight_layout()
    
    classes = [str(np.round(r,1)) for r in np.arange(0.5, 1.6, 0.1)]
    ax[6].bar(classes, energy_dist, alpha=0.5, color="b", label="Energy")
    ax[6].legend(loc=1)
    ax[6].set_xlabel('Classes',fontsize=15) #xlabel
    ax[6].set_ylabel('Softmax Score', fontsize=15) #ylabel
    # pylab.tight_layout()

    pylab.suptitle("Utterance- {}".format(iteration), fontsize=24)
    
    pylab.savefig(os.path.join(hparams.output_directory, "{}.png".format(iteration)))
    pylab.close("all")


def _count_below_alpha(a, h, s, f, alpha=0.05):
    counts = []
    for i in range(len(a)):
        if a[i] < alpha and h[i] < alpha and s[i] < alpha and f[i] < alpha:
            counts.append((i, a[i], h[i], s[i], f[i]))
    return counts


#%%
def test(output_directory, checkpoint_path_rate, 
        checkpoint_path_saliency, hparams, 
        relative_prob, emo_target="angry", valid=True):
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

    model_saliency, model_rate = load_model(hparams)
    criterion = torch.nn.L1Loss()

    test_loader, collate_fn = prepare_dataloaders(hparams, valid=valid)

    # Load checkpoint
    iteration = 0
    model_saliency, _ = load_checkpoint_saliency(checkpoint_path_saliency,
                                                 model_saliency,
                                                 )
    model_rate, _ = load_checkpoint_rate(checkpoint_path_rate, 
                                        model_rate)
    
    WSOLA = WSOLAInterpolationBlockEnergy(win_size=hparams.win_length, 
                                   hop_size=hparams.hop_length,
                                   tolerance=hparams.hop_length)
    OLA = LocalPitchEnergyModification(frame_period=10)

    model_saliency.eval()
    model_rate.eval()

    cunk_array = []
    saliency_loss_array = []
    rate_loss_array = []
    saliency_pred_array = []
    factor_dist_array = []
    factor_array = []
    rate_pred_array = []
    saliency_targ_array = []
    smart_modification_time = []
    greedy_modification_time = []
    
    # ================ MAIN TESTING LOOP! ===================
    for i, batch in enumerate(test_loader):
        start = time.perf_counter()

        x, e, y = batch[0].to("cuda"), batch[1].to("cuda"), batch[2].to("cuda")
        # input_shape should be [#batch_size, 1, #time]

        #%% Sampling the mask only once

        feats, posterior, mask_sample, y_pred = model_saliency(x, e)
        chunked_masks, chunks = get_mask_blocks_inference(mask_sample)
        # feats = feats.repeat(chunked_masks.shape[0],1,1)

        # print("chunked_masks.shape: ", chunked_masks.shape)
        # print("feats.shape: ", feats.shape)
        # print("chunks: ", chunks)

        loss = criterion(y_pred, y)
        saliency_reduced_loss = loss.item()
        
        intent_saliency = intended_saliency(batch_size=1, 
                                            relative_prob=relative_prob)

        # print("intent_saliency.shape: ", intent_saliency.shape)
        (_, rate_distribution,
         pitch_distribution,
         energy_distribution) = model_rate(x.repeat(chunked_masks.shape[0],1,1), 
                                           chunked_masks, 
                                           intent_saliency.repeat(chunked_masks.shape[0],1))

        # print("rate_distribution.shape: ", rate_distribution.shape)

        indices_rate = torch.argmax(rate_distribution, 1)
        # print("indices_rate: ", indices_rate)
        rates = 0.25 + 0.15*indices_rate.reshape(-1,)
        # print("rates: ", rates)
        
        indices_pitch = torch.argmax(pitch_distribution, 1)
        # print("indices_pitch: ", indices_pitch)
        pitches = 0.25 + 0.15*indices_pitch.reshape(-1,)
        # print("pitches: ", pitches)
        
        indices_energy = torch.argmax(energy_distribution, 1)
        # print("indices_pitch: ", indices_pitch)
        energies = 0.25 + 0.15*indices_energy.reshape(-1,)
        # print("pitches: ", pitches)
        
        # index_pitch = torch.multinomial(pitch_distribution[0], 1)
        # pitch = 0.5 + 0.1*index_pitch
        energy_pitch_mod_speech = OLA(factors_pitch=pitches,
                                      factors_energy=energies,
                                      speech=x,
                                      chunks=chunks)

        # modification 1
        mod_speech1, mod_e1, _ = WSOLA(mask=mask_sample[:,:,0], 
                                        rates=rates, 
                                        speech=energy_pitch_mod_speech,
                                        chunks=chunks)
    
        mod_speech1 = mod_speech1.to("cuda")
        mod_e1 = mod_e1.to("cuda")
        _, _, m1, s1 = model_saliency(mod_speech1, mod_e1)
        
        argmax_index = np.argmax(relative_prob)

        mod_speech = mod_speech1
        rate = torch.mean(rates)
        s = s1

        loss = criterion(intent_saliency, s)
        rate_reduced_loss = loss.item()

        feats = feats.detach().squeeze().cpu().numpy()
        x = x.squeeze().cpu().numpy()
        y = y.squeeze().cpu().numpy()
        y_pred = y_pred.squeeze().detach().cpu().numpy()
        s = s.squeeze().detach().cpu().numpy()
        posterior = posterior.squeeze().detach().cpu().numpy()[:,1]
        mask_sample = mask_sample.squeeze().detach().cpu().numpy()[:,1]
        rate_distribution = rate_distribution[0].squeeze().detach().cpu().numpy()
        pitch_distribution = pitch_distribution[0].squeeze().detach().cpu().numpy()
        energy_distribution = energy_distribution[0].squeeze().detach().cpu().numpy()
        mod_speech = mod_speech.squeeze().cpu().numpy()

        # Writing the wav file
        # mod_speech = (mod_speech - np.min(mod_speech)) / (np.max(mod_speech) - np.min(mod_speech))
        # mod_speech = mod_speech - np.mean(mod_speech)
        # sf.write("./output_wavs/{}/{}.wav".format(emo_target, i+1), 
        #             mod_speech.reshape(-1,), 16000)

        #%% Plotting
        saliency_loss_array.append(saliency_reduced_loss)
        rate_loss_array.append(rate_reduced_loss)
        saliency_pred_array.append(y_pred)
        rate_pred_array.append(s)
        saliency_targ_array.append(y)
        factor_dist_array.append(rate_distribution)
        factor_array.append(rate.item())

        # plot_figures(feats, x, mod_speech, posterior, 
        #               mask_sample, y, y_pred, 
        #               rate_distribution,
        #               pitch_distribution,
        #               energy_distribution,
        #               iteration+1, hparams)

        if not math.isnan(saliency_reduced_loss) and not math.isnan(rate_reduced_loss):
            duration = time.perf_counter() - start
            # print("Saliency | Test loss {} {:.6f} {:.2f}s/it".format(
            #     iteration, saliency_reduced_loss, duration))
            # print("Rate    | Test loss {} {:.6f} {:.2f}s/it".format(
            #     iteration, rate_reduced_loss, duration))

        iteration += 1
    
        if iteration >= 250:
            break
    
    print("Saliency | Avg. Loss: {:.3f}".format(np.mean(saliency_loss_array)))
    print("Rate     | Avg. Loss: {:.3f}".format(np.mean(rate_loss_array)))

    return (cunk_array, saliency_targ_array, saliency_pred_array, 
            rate_pred_array, factor_array, factor_dist_array)

#%%
if __name__ == '__main__':
    hparams = create_hparams()

    emo_target = "angry" if len(sys.argv)<2 else sys.argv[1]
    emo_prob_dict = {"angry":[0.0,1.0,0.0,0.0,0.0],
                     "happy":[0.0,0.0,1.0,0.0,0.0],
                     "sad":[0.0,0.0,0.0,0.0,1.0],
                     "fear":[0.0,0.0,0.0,0.0,1.0]}

    emo_model_dict = {"angry":55000, "happy":211000, "sad":154000, "fear":154000}

    ttest_array = []
    count_gr_zero_array = []
    count_flips_array = []
    ckpt_path = hparams.checkpoint_path_inference
    hparams.output_directory = os.path.join(
                                        hparams.output_directory, 
                                        ckpt_path.split("/")[2],
                                        "images_valid_{}".format(emo_target),
                                    )

    if emo_target in emo_model_dict.keys():
        m = emo_model_dict[emo_target]
    
        print("\n \t Current_model: ckpt_{}, Emotion: {}".format(m, emo_target))
        hparams.checkpoint_path_inference = ckpt_path + "_" + str(m)

        if not hparams.output_directory:
            raise FileExistsError('Please specify the output dir.')
        else:
            if not os.path.exists(hparams.output_directory):
                os.makedirs(hparams.output_directory)

        torch.backends.cudnn.enabled = hparams.cudnn_enabled
        torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

        (chunk_array, targ_array, 
         pred_array, rate_array, 
         factor_array, factor_dist_array) = test(
                                                hparams.output_directory,
                                                hparams.checkpoint_path_inference,
                                                hparams.checkpoint_path_saliency,
                                                hparams,
                                                emo_prob_dict[emo_target],
                                                emo_target=emo_target,
                                                valid=False,
                                            )
        
        pred_array = np.asarray(pred_array)
        targ_array = np.asarray(targ_array)
        rate_array = np.asarray(rate_array)

        #%% Checking difference in predictions
        index = np.argmax(emo_prob_dict[emo_target])
        saliency_diff = (rate_array[:,index] - pred_array[:,index]) / (pred_array[:,index] + 1e-10)
        count = len(np.where(np.asarray(saliency_diff)>0)[0])
        ttest = scistat.ttest_1samp(a=saliency_diff, popmean=0, alternative="greater")
        print("1 sided T-test result (p-value): {} and count greater zero: {}".format(ttest[1], count))
        ttest_array.append(ttest[1])
        count_gr_zero_array.append(count)
        
        idx = np.where(saliency_diff>0)[0]
        count_flips = 0
        count_neutral = 0
        count_neutral_flips = 0
        indices_flips = []
        for i in range(targ_array.shape[0]):
            # if np.argmax(pred_array[i,:])!=index and np.argmax(rate_array[i,:])==index:
            #     count_flips += 1
            #     indices_flips.append(i+1)
            
            # if (pred_array[i,index] <= 0.25) and (index in np.argsort(rate_array[i,:])[-2:]):
            #     count_flips += 1
            #     indices_flips.append(i+1)
            
            if (index not in np.argsort(pred_array[i,:])[-2:]) and (index in np.argsort(rate_array[i,:])[-2:]):
                count_flips += 1
                indices_flips.append(i+1)
            
            if np.argmax(pred_array[i,:])==0 and np.argmax(rate_array[i,:])==index:
                count_neutral_flips += 1

            if np.argmax(targ_array[i,:])==0:
                count_neutral += 1
        
        count_flips_array.append(count_flips)
        print("Flip Counts: {} and Neutral Flips: {}".format(count_flips, count_neutral_flips))
        print("Total neutral: {}".format(count_neutral))
        
        # joblib.dump({"ttest_scores": ttest_array, 
        #             "count_scores": count_gr_zero_array,
        #             "count_flips": count_flips_array}, os.path.join(hparams.output_directory,
        #                                                         "ttest_scores_argmax.pkl"))


        # joblib.dump({"indices": indices_flips}, 
        #             "./output_wavs/{}/indices.pkl".format(emo_target))
        
        # print("average difference: ", np.mean(rate_array - pred_array, axis=0))
        
        #%%
        count_not_targ = 0
        for t in pred_array:
            if index not in list(np.argsort(t)[-2:]):
                count_not_targ += 1
        
        print("Target not in top 2 predictions: ", count_not_targ)
        print("Flipping ratio: ", count_flips/count_not_targ)
        #%%
        idx = np.where(saliency_diff>0)[0]
        # idx = np.arange(0, len(rate_array))
        diff_n = rate_array[idx, 0] - pred_array[idx, 0]
        diff_a = rate_array[idx, 1] - pred_array[idx, 1]
        diff_h = rate_array[idx, 2] - pred_array[idx, 2]
        diff_s = rate_array[idx, 4] - pred_array[idx, 4]
        # diff_f = rate_array[idx, 4] - pred_array[idx, 4]
        pylab.figure()
        ax = pylab.subplot(111)
        pylab.violinplot([diff_n, diff_a, diff_h, diff_s], 
                         positions=[0,1,2,3], vert=True, showmedians=True)
        ax.set_xticks([0,1,2,3])
        ax.set_xticklabels(["Neutral", "Angry", "Happy", "Sad"])
        # pylab.title("Target- {}".format(emo_target))
        # pylab.savefig("./output_wavs/AC_energy_{}_difference_plot.png".format(emo_target))
       

























