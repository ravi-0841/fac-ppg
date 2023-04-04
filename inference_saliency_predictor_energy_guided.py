#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 14:28:20 2023

@author: ravi
"""


import os
import time
import math
import torch
import pylab
import numpy as np
import seaborn as sns
from scipy.signal import medfilt
from torch.utils.data import DataLoader
from saliency_predictor_energy_guided import MaskedSaliencePredictor
from on_the_fly_augmentor_raw_voice_mask import OnTheFlyAugmentor, acoustics_collate_raw
from src.common.loss_function import (MaskedSpectrogramL1LossReduced,
                                        ExpectedKLDivergence,
                                        VecExpectedKLDivergence, 
                                        SparsityKLDivergence,
                                    )
from src.common.utils import (median_mask_filtering, 
                              refining_mask_sample,
                              )
from src.common.hparams_onflyenergy import create_hparams
from pprint import pprint


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
    model = MaskedSaliencePredictor(hparams.temp_scale).cuda()
    return model


def load_checkpoint(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    # optimizer.load_state_dict(checkpoint_dict['optimizer'])
    # learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, iteration


def plot_figures(waveform, feats, posterior, mask, y, y_pred, iteration, hparams):
    
    mask_thresh = np.zeros((len(mask),))
    mask_thresh[np.where(mask>0)[0]] = 1

    # Plotting details
    pylab.xticks(fontsize=18)
    pylab.yticks(fontsize=18)
    fig, ax = pylab.subplots(4, 1, figsize=(32, 18))

    ax[0].plot(waveform.reshape(-1,), linewidth=1.5, color='r')
    ax[0].set_xlabel('Time',fontsize = 20) #xlabel
    ax[0].set_ylabel('Magnitude', fontsize = 20) #ylabel
    # pylab.tight_layout()
    
    ax[1].imshow(np.log10(np.abs(feats) + 1e-10), aspect="auto", origin="lower",
                   interpolation='none')
    ax[1].plot(251*mask_thresh, "w", linewidth=4.0)
    ax[1].set_xlabel('Time',fontsize = 20) #xlabel
    ax[1].set_ylabel('Frequency', fontsize = 20) #ylabel
    # pylab.tight_layout()

    ax[2].plot(posterior, linewidth=2.5, color='b')
    ax[2].plot(mask_thresh, linewidth=2.5, color='g')
    ax[2].set_xlabel('Time',fontsize = 20) #xlabel
    ax[2].set_ylabel('Probability', fontsize = 20) #ylabel
    # pylab.tight_layout()
    
    classes = ["neu", "ang", "hap", "sad", "fea"]
    ax[3].bar(classes, y, alpha=0.5, label="target")
    ax[3].bar(classes, y_pred, alpha=0.5, label="pred")
    ax[3].legend(loc=1)
    ax[3].set_xlabel('Classes',fontsize = 20) #xlabel
    ax[3].set_ylabel('Softmax Score', fontsize = 20) #ylabel
    # pylab.tight_layout()

    pylab.suptitle("Utterance- {}".format(iteration), fontsize=24)
    
    pylab.savefig(os.path.join(hparams.output_directory, "{}.png".format(iteration)))
    pylab.close("all")


def multi_sampling(model, x, y, criterion, num_samples=5):
    
    assert num_samples >= 3, "Sample at least 3 times"
    
    mask_samples = []
    posterior, mask, y_pred = model(x)
    loss = criterion(y_pred, y)
    reduced_loss = loss.item()

    y = y.squeeze().cpu().numpy()
    posterior = posterior.squeeze().detach().cpu().numpy()[:,1]
    mask = mask.squeeze().detach().cpu().numpy()[:,1]
    y_pred = y_pred.squeeze().detach().cpu().numpy()
    
    # mask_samples.append(refining_mask_sample(mask)[1])
    mask_samples.append(medfilt(mask, kernel_size=3))
    
    for _ in range(num_samples-1):
        _, m, _ = model(x)
        m = m.squeeze().detach().cpu().numpy()[:,1]
        # mask_samples.append(refining_mask_sample(m)[1])
        mask_samples.append(medfilt(m, kernel_size=3))
    
    mask_intersect = np.multiply(np.logical_and(mask_samples[0], mask_samples[1]), 1)
    for i in range(2, num_samples):
        mask_intersect = np.multiply(np.logical_and(mask_intersect, mask_samples[i]), 1)
    
    for _ in range(7): #7
        mask_intersect = medfilt(mask_intersect, kernel_size=7) #7
    
    x = x.squeeze().cpu().numpy()
    return x, y, y_pred, posterior, mask_intersect, reduced_loss


def random_mask_thresholding(mask, threshold=5):
    start_pointer = None
    end_pointer = None

    for i, m in enumerate(mask):
        if m > 0 and start_pointer is None:
            start_pointer = i
            end_pointer = None
        
        elif m < 1 and start_pointer is not None:
            end_pointer = i-1
    
            if (end_pointer - start_pointer + 1) < threshold:
                mask[start_pointer:end_pointer+1] = 0
                # break
            
            start_pointer = None

    return mask        


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

    # max_val = y_pred[np.flip(np.argsort(y_pred))[k]]
    # pred_idxs = [index for index, value in enumerate(y_pred) if value == max_val]

    # list_intersection = intersection(targ_idxs, pred_idxs)

    # if len(list_intersection)>0:
    #     return 1
    # else:
    #     return 0

    # if np.flip(np.argsort(y_true))[0] == np.flip(np.argsort(y_pred))[k]:
    #     return 1
    # else:
    #     return 0


def compute_MI(marg1, marg2, joint):
    mi = 0
    for r in range(joint.shape[0]):
        for c in range(joint.shape[1]):
            mi += (joint[r,c]*np.log(joint[r,c]/(marg1[r]*marg2[c])))         
    return mi


def test(output_directory, checkpoint_path, hparams, valid=True):
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

    model = load_model(hparams)
    criterion = torch.nn.L1Loss()

    test_loader, collate_fn = prepare_dataloaders(hparams, valid=valid)

    # Load checkpoint
    iteration = 0
    model, _ = load_checkpoint(checkpoint_path, model)

    model.eval()

    cunk_array = []
    loss_array = []
    pred_array = []
    targ_array = []
    
    # ================ MAIN TESTING LOOP! ===================
    for i, batch in enumerate(test_loader):
        start = time.perf_counter()

        x, e, y = batch[0].to("cuda"), batch[1].to("cuda"), batch[2].to("cuda")
        # input_shape should be [#batch_size, #freq_channels, #time]

        #%% Sampling the mask only once
        
        feats, posterior, mask_sample, y_pred = model(x, e)
        loss = criterion(y_pred, y)
        reduced_loss = loss.item()
        loss_array.append(reduced_loss)

        feats = feats.squeeze().detach().cpu().numpy()
        x = x.squeeze().cpu().numpy()
        y = y.squeeze().cpu().numpy()
        y_pred = y_pred.squeeze().detach().cpu().numpy()
        posterior = posterior.squeeze().detach().cpu().numpy()[:,1]
        mask_sample = mask_sample.squeeze().detach().cpu().numpy()[:,1]

        #%% Plotting

        loss_array.append(reduced_loss)
        pred_array.append(y_pred)
        targ_array.append(y)

        # plot_figures(x, feats, posterior, mask_sample, y, 
        #               y_pred, iteration+1, hparams)

        # if not math.isnan(reduced_loss):
        #     duration = time.perf_counter() - start
            # print("Test loss {} {:.6f} {:.2f}s/it".format(
            #     iteration, reduced_loss, duration))

        iteration += 1
    
    print("Avg. Loss: {:.3f}".format(np.mean(loss_array)))
    
    return cunk_array, targ_array, pred_array


if __name__ == '__main__':
    hparams = create_hparams()
    
    ckpt_path = hparams.checkpoint_path_inference
    hparams.output_directory = os.path.join(
                                        hparams.output_directory, 
                                        ckpt_path.split("/")[2],
                                        "images_valid",
                                        )

    if not hparams.output_directory:
        raise FileExistsError('Please specify the output dir.')
    else:
        if not os.path.exists(hparams.output_directory):
            os.makedirs(hparams.output_directory)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    chunk_array, targ_array, pred_array = test(
                                                hparams.output_directory,
                                                hparams.checkpoint_path_inference,
                                                hparams,
                                                valid=False,
                                                )
    
    pred_array = np.asarray(pred_array)
    targ_array = np.asarray(targ_array)
    
    top_1 = [best_k_class_metric(t, p, k=0) for (t, p) in zip(targ_array, pred_array)]
    top_2 = [best_k_class_metric(t, p, k=1) for (t, p) in zip(targ_array, pred_array)]
    
    print("Top-1 Accuracy is: {}".format(np.round(np.sum(top_1)/len(top_1),4)))
    print("Top-2 Accuracy is: {}".format(np.round((np.sum(top_1) + np.sum(top_2))/len(top_1),4)))

    #%% Energy Posterior correlation
    # pylab.figure(figsize=(10,10)), sns.histplot(corr_array[:,0], bins=30, kde=True)
    # pylab.title("Correlation between posterior and energy contour, median- {}".format(
    #                                                     np.round(np.median(corr_array[:,0]), 2)
    #                                                     ))
    # pylab.savefig(os.path.join(hparams.output_directory, "correlation_energy.png"))

    # pylab.figure(figsize=(10,10)), sns.histplot(corr_array[:,1], bins=30, kde=True)
    # pylab.title("Correlation between posterior and energy contour gradient, median- {}".format(
    #                                                     np.round(np.median(corr_array[:,1]), 2)
    #                                                     ))
    # pylab.savefig(os.path.join(hparams.output_directory, "correlation_energy_gradient.png"))
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
    # pylab.title("Mutual Information distribution")
    # pylab.savefig(os.path.join(hparams.output_directory, "MI_density.png"))
    # pylab.close("all")
    




























