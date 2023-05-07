#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 14:28:20 2023

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
                            shuffle=False,
                            sampler=None,
                            batch_size=1,
                            drop_last=False,
                            collate_fn=collate_fn,
                            )
    return test_loader, collate_fn


def load_model(hparams):
    model_saliency = MaskedRateModifier(hparams.temp_scale).cuda()
    model_rate = RatePredictor(temp_scale=1.0).cuda()
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
def test(output_directory, checkpoint_path_rate, 
        checkpoint_path_saliency, hparams, 
        relative_prob, valid=True):
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
    
    WSOLA = WSOLAInterpolationEnergy(win_size=hparams.win_length, 
                                   hop_size=hparams.hop_length,
                                   tolerance=hparams.hop_length)

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
    
    # ================ MAIN TESTING LOOP! ===================
    for i, batch in enumerate(test_loader):
        start = time.perf_counter()

        try:
            x, e, y = batch[0].to("cuda"), batch[1].to("cuda"), batch[2].to("cuda")
            # input_shape should be [#batch_size, 1, #time]
    
            #%% Sampling the mask only once
    
            feats, posterior, mask_sample, y_pred = model_saliency(x, e)
            loss = criterion(y_pred, y)
            saliency_reduced_loss = loss.item()
            
            intent_saliency = intended_saliency(batch_size=1, 
                                                relative_prob=relative_prob)
    
            rate_distribution = model_rate(feats, mask_sample, intent_saliency)
            # index = torch.multinomial(rate_distribution, 1)
            index = torch.argmax(rate_distribution, 1)
            rate = 0.5 + 0.1*index # 0.2*index
            mod_speech, mod_e, _ = WSOLA(mask=mask_sample[:,:,0], 
                                        rate=rate, speech=x)
        
            mod_speech = mod_speech.to("cuda")
            mod_e = mod_e.to("cuda")
            _, _, m, s = model_saliency(mod_speech, mod_e)
            
            loss = criterion(intent_saliency, s)
            rate_reduced_loss = loss.item()
    
            feats = feats.detach().squeeze().cpu().numpy()
            x = x.squeeze().cpu().numpy()
            y = y.squeeze().cpu().numpy()
            y_pred = y_pred.squeeze().detach().cpu().numpy()
            s = s.squeeze().detach().cpu().numpy()
            posterior = posterior.squeeze().detach().cpu().numpy()[:,1]
            mask_sample = mask_sample.squeeze().detach().cpu().numpy()[:,1]
            rate_distribution = rate_distribution.squeeze().detach().cpu().numpy()
            mod_speech = mod_speech.squeeze().cpu().numpy()
    
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
            #               iteration+1, hparams)
    
            if not math.isnan(saliency_reduced_loss) and not math.isnan(rate_reduced_loss):
                duration = time.perf_counter() - start
                # print("Saliency | Test loss {} {:.6f} {:.2f}s/it".format(
                #     iteration, saliency_reduced_loss, duration))
                # print("Rate    | Test loss {} {:.6f} {:.2f}s/it".format(
                #     iteration, rate_reduced_loss, duration))
    
            iteration += 1
        
        # if iteration >= 100:
        #     break
        
        except Exception as ex:
            print(ex)
    
    print("Saliency | Avg. Loss: {:.3f}".format(np.mean(saliency_loss_array)))
    print("Rate     | Avg. Loss: {:.3f}".format(np.mean(rate_loss_array)))

    return (cunk_array, saliency_targ_array, saliency_pred_array, 
            rate_pred_array, factor_array, factor_dist_array)

#%%
if __name__ == '__main__':
    hparams = create_hparams()

    emo_target = "fear" #sys.argv[1] #"angry"
    emo_prob_dict = {"angry":[0.0,1.0,0.0,0.0,0.0],
                     "happy":[0.0,0.0,1.0,0.0,0.0],
                     "sad":[0.0,0.0,0.0,1.0,0.0],
                     "fear":[0.0,0.0,0.0,0.0,1.0]}

    ttest_array = []
    ckpt_path = hparams.checkpoint_path_inference
    hparams.output_directory = os.path.join(
                                        hparams.output_directory, 
                                        ckpt_path.split("/")[2],
                                        "images_valid_{}".format(emo_target),
                                    )

    # for m in range(85500, 85700, 750):
    for m in range(76500, 77000, 750):
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
                                                valid=True,
                                            )
        
        pred_array = np.asarray(pred_array)
        targ_array = np.asarray(targ_array)
        rate_array = np.asarray(rate_array)
        
        top_1 = [best_k_class_metric(t, p, k=0) for (t, p) in zip(targ_array, pred_array)]
        top_2 = [best_k_class_metric(t, p, k=1) for (t, p) in zip(targ_array, pred_array)]
        
        print("Top-1 Accuracy is: {}".format(np.round(np.sum(top_1)/len(top_1),4)))
        print("Top-2 Accuracy is: {}".format(np.round((np.sum(top_1) + np.sum(top_2))/len(top_1),4)))

        #%% Checking difference in predictions
        index = np.argmax(emo_prob_dict[emo_target])
        saliency_diff = (rate_array[:,index] - pred_array[:,index]) / (pred_array[:,index] + 1e-10)
        ttest = scistat.ttest_1samp(a=saliency_diff, popmean=0, alternative="greater")
        print("1 sided T-test result (p-value): {}".format(ttest[1]))
        ttest_array.append(ttest[1])
        # joblib.dump({"ttest_scores": ttest_array}, os.path.join(hparams.output_directory,
        #                                                         "ttest_scores.pkl"))

        # pylab.figure(), pylab.hist(saliency_diff, label="difference")
        # pylab.savefig(os.path.join(hparams.output_directory, "histplot_{}.png".format(emo_target)))
        # pylab.close("all")
        
        # pylab.figure(), pylab.plot([x for x in range(1000, 188000, 1000)], ttest_array)
        # pylab.title(emo_target)
        # pylab.savefig(os.path.join(hparams.output_directory, "ttest_scores.png"))
        # pylab.close("all")

        #%% Joint density plot and MI
        # epsilon = 1e-3
        # corn_mat = np.zeros((5,5))
        # for (t,p) in zip(targ_array, pred_array):
        #     for et in range(5):
        #         for ep in range(5):
        #             if t[et]>epsilon and p[ep]>epsilon:
        #                 corn_mat[ep, et] += 1
        # corn_mat = corn_mat / np.sum(corn_mat)
        # x = np.arange(0, 6, 1)
        # y = np.arange(0, 6, 1)
        # x_center = 0.5 * (x[:-1] + x[1:])
        # y_center = 0.5 * (y[:-1] + y[1:])
        # X, Y = np.meshgrid(x_center, y_center)
        # plot = pylab.pcolormesh(x, y, corn_mat, cmap='RdBu', shading='flat')
        # cset = pylab.contour(X, Y, corn_mat, cmap='gray')
        # pylab.clabel(cset, inline=True)
        # pylab.colorbar(plot)
        # pylab.title("Joint density estimate")
        # pylab.savefig(os.path.join(hparams.output_directory, "joint_density_plot.png"))
        # pylab.close("all")

        # # Mutual Info
        # mi_array = [compute_MI(p+1e-10,t+1e-10,corn_mat) for (p,t) in zip(pred_array, targ_array)]
        # sns.histplot(mi_array, bins=30, kde=True)
        # print("MI value: {}".format(np.mean(mi_array)))
        # pylab.title("Mutual Information distribution")
        # pylab.savefig(os.path.join(hparams.output_directory, "MI_density.png"))
        # pylab.close("all")

    #%%
    # x = np.arange(750, 150750, 750)
    # angry_scores = joblib.load("/home/ravi/RockFish/fac-ppg/masked_predictor_output/OnlyRate_entropy_0.07_exploit_0.15_TD_RL_lr_1e-7_evm_bs_4/images_valid_angry/ttest_scores.pkl")["ttest_scores"]
    # happy_scores = joblib.load("/home/ravi/RockFish/fac-ppg/masked_predictor_output/OnlyRate_entropy_0.07_exploit_0.15_TD_RL_lr_1e-7_evm_bs_4/images_valid_happy/ttest_scores.pkl")["ttest_scores"]
    # sad_scores = joblib.load("/home/ravi/RockFish/fac-ppg/masked_predictor_output/OnlyRate_entropy_0.07_exploit_0.15_TD_RL_lr_1e-7_evm_bs_4/images_valid_sad/ttest_scores.pkl")["ttest_scores"]
    # fear_scores = joblib.load("/home/ravi/RockFish/fac-ppg/masked_predictor_output/OnlyRate_entropy_0.07_exploit_0.15_TD_RL_lr_1e-7_evm_bs_4/images_valid_fear/ttest_scores.pkl")["ttest_scores"]
    
    # pylab.figure()
    # pylab.plot(x, angry_scores, "o", label="angry")
    # pylab.plot(x, happy_scores, "o", label="happy")
    # pylab.plot(x, sad_scores, "o", label="sad")
    # pylab.plot(x, fear_scores, "o", label="fear")
    # pylab.plot(x, [0.1]*len(angry_scores), label="baseline1")
    # pylab.plot(x, [0.07]*len(angry_scores), label="baseline2")
    # pylab.plot(x, [0.05]*len(angry_scores), label="baseline3")
    # pylab.legend()




























