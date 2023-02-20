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

from pytorch_grad_cam import GradCAM
from scipy.signal import medfilt
from torch.utils.data import DataLoader
from saliency_predictor_2D import SaliencyPredictor
from on_the_fly_augmentor import OnTheFlyAugmentor, acoustics_collate
from src.common.loss_function import (MaskedSpectrogramL1LossReduced,
                                        ExpectedKLDivergence,
                                        VecExpectedKLDivergence, 
                                        SparsityKLDivergence,
                                    )
from src.common.utils import (median_mask_filtering, 
                              refining_mask_sample,
                              )
from src.common.hparams_onflyaugmentor import create_hparams
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

    collate_fn = acoustics_collate
    
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
    model = SaliencyPredictor(hparams.temp_scale).cuda()
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


def plot_gray_cam(spectrogram, gray_cam, y, y_pred, iteration, hparams):

    # Plotting details
    pylab.xticks(fontsize=18)
    pylab.yticks(fontsize=18)
    fig, ax = pylab.subplots(3, 1, figsize=(24, 15))
    
    ax[0].imshow(np.log10(spectrogram[0] + 1e-10), aspect="auto", origin="lower",
                   interpolation='none')
    ax[0].set_xlabel('Time',fontsize = 20) #xlabel
    ax[0].set_ylabel('Frequency', fontsize = 20) #ylabel
    # pylab.tight_layout()
    
    ax[1].imshow(gray_cam[0], aspect="auto", origin="lower",
                   interpolation='none')
    ax[1].set_xlabel('Time',fontsize = 20) #xlabel
    ax[1].set_ylabel('Frequency', fontsize = 20) #ylabel
    # pylab.tight_layout()
    
    classes = ["neu", "ang", "hap", "sad", "fea"]
    ax[2].bar(classes, y[0], alpha=0.5, label="target")
    ax[2].bar(classes, y_pred[0], alpha=0.5, label="pred")
    ax[2].legend(loc=1)
    ax[2].set_xlabel('Classes',fontsize = 20) #xlabel
    ax[2].set_ylabel('Softmax Score', fontsize = 20) #ylabel
    # pylab.tight_layout()

    pylab.suptitle("Utterance- {}".format(iteration), fontsize=24)
    
    pylab.savefig(os.path.join(hparams.output_directory, "{}.png".format(iteration)))
    pylab.close("all")


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
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)
    criterion = torch.nn.L1Loss()

    test_loader, collate_fn = prepare_dataloaders(hparams, valid=valid)

    # Load checkpoint
    iteration = 1
    model, _, _, _ = load_checkpoint(checkpoint_path, model, optimizer)

    model.eval()
    
    grad_cam = GradCAM(model=model, target_layers=[model.conv1_enc.conv1], 
                        use_cuda=True)
    grad_cam.model.train()

    loss_array = []
    pred_array = []
    targ_array = []
    
    # ================ MAIN TESTING LOOP! ===================
    for i, batch in enumerate(test_loader):
        start = time.perf_counter()

        x, y, _ = batch[0].to("cuda"), batch[1].to("cuda"), batch[2]
        # input_shape should be [#batch_size, #freq_channels, #time]

        #%% Sampling masks multiple times for same utterance
        gray_cam = grad_cam(input_tensor=x, targets=None)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        reduced_loss = loss.item()
        
        #%% Plotting
        x = x.cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        
        loss_array.append(reduced_loss)
        pred_array.append(y_pred)
        targ_array.append(y)
        
        plot_gray_cam(x, gray_cam, y, y_pred, iteration, hparams)

        if not math.isnan(reduced_loss):
            duration = time.perf_counter() - start
            print("Test loss {} {:.6f} {:.2f}s/it".format(
                iteration, reduced_loss, duration))

        iteration += 1
        
        # if i == 30:
        #     break
    
    print("Avg. Loss: {:.3f}".format(np.mean(loss_array)))
    
    return targ_array, pred_array


if __name__ == '__main__':
    hparams = create_hparams()

    hparams.output_directory = os.path.join(
                                        hparams.output_directory, 
                                        "2D_{}_{}".format(
                                            hparams.temp_scale,
                                            hparams.extended_desc,
                                        ),
                                        "images_test_2"
                                    )

    if not hparams.output_directory:
        raise FileExistsError('Please specify the output dir.')
    else:
        if not os.path.exists(hparams.output_directory):
            os.makedirs(hparams.output_directory)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    targ_array, pred_array = test(
                                hparams.output_directory,
                                hparams.checkpoint_path,
                                hparams,
                                valid=False,
                            )
    
    pred_array = np.vstack(pred_array)
    targ_array = np.vstack(targ_array)
    
    top_1 = [best_k_class_metric(t, p, k=0) for (t, p) in zip(targ_array, pred_array)]
    top_2 = [best_k_class_metric(t, p, k=1) for (t, p) in zip(targ_array, pred_array)]
    
    print("Top-1 Accuracy is: {}".format(np.round(np.sum(top_1)/len(top_1),2)))
    print("Top-2 Accuracy is: {}".format(np.round((np.sum(top_1) + np.sum(top_2))/len(top_1),2)))
    




























