#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 14:04:08 2023

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
import joblib

from scipy.signal import medfilt
from torch.utils.data import DataLoader
from saliency_predictor_energy_guided_RL import MaskedRateModifier, RatePredictor
from on_the_fly_augmentor_raw_voice_mask import OnTheFlyAugmentor, acoustics_collate_raw
from src.common.loss_function import (MaskedSpectrogramL1LossReduced,
                                        ExpectedKLDivergence,
                                        VecExpectedKLDivergence, 
                                        SparsityKLDivergence,
                                    )
from src.common.utils import (median_mask_filtering, 
                              refining_mask_sample,
                              )
from src.common.hparams_onflyenergy_rate import create_hparams
from src.common.interpolation_block import (WSOLAInterpolation, 
                                            WSOLAInterpolationEnergy,
                                            BatchWSOLAInterpolation,
                                            BatchWSOLAInterpolationEnergy)
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
    model_rate = RatePredictor(temp_scale=0.2).cuda()
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
                 mask, y, y_pred, rate_dist, iteration, hparams):
    
    mask_thresh = np.zeros((len(mask), ))
    mask_thresh[np.where(mask>0)[0]] = 1

    # Plotting details
    pylab.xticks(fontsize=18)
    pylab.yticks(fontsize=18)
    fig, ax = pylab.subplots(5, 1, figsize=(30, 20))
    
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
    ax[4].bar(classes, rate_dist, alpha=0.5, color="r", label="pred")
    ax[4].legend(loc=1)
    ax[4].set_xlabel('Classes',fontsize=15) #xlabel
    ax[4].set_ylabel('Softmax Score', fontsize=15) #ylabel
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


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def best_k_class_metric(y_true, y_pred, k=0):
    # k is either 0 or 1
    max_val = np.max(y_true)
    targ_idxs = [index for index, value in enumerate(y_true) if value == max_val]
    
    if k == 0:
        chek_idx = np.flip(np.argsort(y_pred))[0]
        if chek_idx in targ_idxs:
            return 1
        else:
            return 0
    else:
        sorted_idx = np.flip(np.argsort(y_pred))
        ignore_idx = sorted_idx[0]
        chek_idx = sorted_idx[k]
        if (chek_idx in targ_idxs) and (ignore_idx not in targ_idxs):
            return 1
        else:
            return 0


def compute_MI(marg1, marg2, joint):
    mi = 0
    for r in range(joint.shape[0]):
        for c in range(joint.shape[1]):
            mi += (joint[r,c]*np.log(joint[r,c]/(marg1[r]*marg2[c])))         
    return mi

#%%
def test(checkpoint_path_saliency, hparams, valid=True):
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

    test_loader, collate_fn = prepare_dataloaders(hparams, valid=valid)

    # Load checkpoint
    model_saliency, _ = load_checkpoint_saliency(checkpoint_path_saliency,
                                                 model_saliency,
                                                 )
    
    WSOLA = WSOLAInterpolationEnergy(win_size=hparams.win_length, 
                                   hop_size=hparams.hop_length,
                                   tolerance=hparams.hop_length)

    model_saliency.eval()
    
    # ================ MAIN TESTING LOOP! ===================
    for i, batch in enumerate(test_loader):
        try:
            x, e = batch[0].to("cuda"), batch[1].to("cuda")
            # input_shape should be [#batch_size, 1, #time]
    
            #%% Generating all possible modifications
    
            feats, posterior, mask_sample, y_pred = model_saliency(x, e)
            rates = np.round(np.arange(0.5, 2.1, 0.1), 1)
            s_s = []
            for r in list(rates):
                r = torch.Tensor([r]).to("cuda")
                mod_speech, mod_e, _ = WSOLA(mask=mask_sample[:,:,0], 
                                            rate=r, speech=x)
        
                mod_speech = mod_speech.to("cuda")
                mod_e = mod_e.to("cuda")
                _, _, m, s = model_saliency(mod_speech, mod_e)
                s_s.append(s.detach().cpu().numpy().reshape(-1,))

            y_pred = y_pred.squeeze().detach().cpu().numpy().reshape(-1,)
    
            #%% Plotting
            if not os.path.exists(os.path.join("/home/ravi/Desktop/visualization", str(i))):    
                os.makedirs(os.path.join("/home/ravi/Desktop/visualization", str(i)))
            X = ["N", "A", "H", "S", "F"]
            rate_str = ["05", "06", "07", "08", "09", "10", "11", "12", "13", 
                        "14", "15", "16", "17", "18", "19", "20"]
            for idx,s in enumerate(s_s):
                pylab.figure()
                pylab.bar(X, y_pred, label="original", alpha=0.5)
                pylab.bar(X, s, label="modified", alpha=0.5), pylab.legend(loc=1)
                pylab.suptitle("modification rate - {}".format(rates[idx]))
                pylab.savefig("/home/ravi/Desktop/visualization/{}/modif_rate_{}.png".format(str(i), rate_str[idx]))
                pylab.close()
            
            if i >= 10:
                break
            
        except Exception as ex:
            print(ex)

    return None

#%%
if __name__ == '__main__':
    hparams = create_hparams()

    emo_target = "angry"

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    test(
        hparams.checkpoint_path_saliency,
        hparams,
        valid=True,
        )
        
