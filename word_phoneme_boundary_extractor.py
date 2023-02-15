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
import textgrid

from pprint import pprint
from scipy.signal import medfilt
from torch.utils.data import DataLoader
from saliency_predictor import SaliencyPredictor
from on_the_fly_augmentor import OnTheFlyAugmentor, acoustics_collate
from src.common.hparams_onflyaugmentor import create_hparams
from src.common.loss_function import (MaskedSpectrogramL1LossReduced,
                                        ExpectedKLDivergence,
                                        VecExpectedKLDivergence, 
                                        SparsityKLDivergence,
                                    )
from src.common.utils import (median_mask_filtering, 
                              refining_mask_sample,
                              )
from dialog_path_forForcedAligner import (format_audio_text,
                                          prepare_dialog_lookup,
                                          )
from inference_saliency_predictor import (load_model,
                                          load_checkpoint,
                                          multi_sampling,
                                          )


def prepare_dataloaders_and_lookup_dict(hparams, valid=True):
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
    
    lookup_dict = prepare_dialog_lookup()

    return lookup_dict, testset, test_loader, collate_fn


def plot_figures(spectrogram, posterior, mask, y, y_pred, iteration, hparams):
    # Plotting details
    pylab.xticks(fontsize=18)
    pylab.yticks(fontsize=18)
    fig, ax = pylab.subplots(4, 1, figsize=(32, 18))
    
    energy = np.sum(spectrogram**2, axis=0)
    ax[0].plot(energy, linewidth=2.5, color='r')
    ax[0].set_xlabel('Time',fontsize = 20) #xlabel
    ax[0].set_ylabel('Energy', fontsize = 20) #ylabel
    # pylab.tight_layout()
    
    ax[1].imshow(np.log10(spectrogram + 1e-10), aspect="auto", origin="lower",
                   interpolation='none')
    ax[1].plot(151*mask, "w", linewidth=4.0)
    ax[1].set_xlabel('Time',fontsize = 20) #xlabel
    ax[1].set_ylabel('Frequency', fontsize = 20) #ylabel
    # pylab.tight_layout()

    ax[2].plot(posterior, linewidth=2.5, color='k')
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

    correlation = np.corrcoef(energy, posterior)[0,1]

    pylab.suptitle("Utterance- {}, correlation (energy/posterior)- {}".format(iteration, 
                    np.round(correlation, 2)), 
                    fontsize=24)
    
    pylab.savefig(os.path.join(hparams.output_directory, "{}.png".format(iteration)))
    pylab.close("all")
    return correlation


def get_phones_and_words(textgrid_object):
    phone_tiers = textgrid_object.getList("phone")[0]
    word_tiers = textgrid_object.getList("word")[0]
    
    phone_info = []
    word_info = []
    
    for p in phone_tiers:
        phone_info.append((p.mark, p.minTime, p.maxTime))
    
    for w in word_tiers:
        word_info.append((w.mark, w.minTime, w.maxTime))
    
    return phone_info, word_info


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

    (lookup_dict, testset, 
     test_loader, _) = prepare_dataloaders_and_lookup_dict(hparams, valid=valid)

    # Load checkpoint
    iteration = 0
    model, _, _, _ = load_checkpoint(checkpoint_path, model, optimizer)

    model.eval()

    cunk_array = []
    loss_array = []
    pred_array = []
    targ_array = []
    corr_array = []
    text_grid_array = []
    
    # ================ MAIN TESTING LOOP! ===================
    for i in range(len(testset)):
        start = time.perf_counter()
        
        x, _, y, _ = testset[i]
        x, y = x.unsqueeze(dim=0).to("cuda"), y.to("cuda")
        # input_shape should be [#batch_size, #freq_channels, #time]

        #%% Sampling masks multiple times for same utterance
        
        (x, y, y_pred, posterior, 
         mask, reduced_loss) = multi_sampling(model, x, y, criterion)

        text_grid = format_audio_text(data_object=testset, 
                                                 index=i, 
                                                 lookup_dict=lookup_dict,
                                                 )
        phones, words = get_phones_and_words(text_grid)

        loss_array.append(reduced_loss)
        pred_array.append(y_pred)
        targ_array.append(y)
        text_grid_array.append(text_grid)
        
        #%% Plotting
        # corr_array.append(plot_figures(x, posterior, mask, y, 
        #                                y_pred, iteration+1, hparams))

        if not math.isnan(reduced_loss):
            duration = time.perf_counter() - start
            print("Test loss {} {:.6f} {:.2f}s/it".format(
                iteration, reduced_loss, duration))

        iteration += 1
    
    print("Avg. Loss: {:.3f}".format(np.mean(loss_array)))
    
    return cunk_array, targ_array, pred_array, text_grid_array


if __name__ == '__main__':
    hparams = create_hparams()

    hparams.output_directory = os.path.join(
                                        hparams.output_directory, 
                                        "libri_{}_{}_{}_{}_{}".format(
                                            hparams.lambda_prior_KL,
                                            hparams.lambda_predict,
                                            hparams.lambda_sparse_KL,
                                            hparams.temp_scale,
                                            hparams.extended_desc,
                                        ),
                                        "images_destroy"
                                    )

    if not hparams.output_directory:
        raise FileExistsError('Please specify the output dir.')
    else:
        if not os.path.exists(hparams.output_directory):
            os.makedirs(hparams.output_directory)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    _, targ_array, pred_array, grid_array = test(
                                                hparams.output_directory,
                                                hparams.checkpoint_path,
                                                hparams,
                                                valid=True,
                                            )
    
    pred_array = np.asarray(pred_array)
    targ_array = np.asarray(targ_array)





























