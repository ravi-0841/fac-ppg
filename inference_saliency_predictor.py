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
from scipy.signal import medfilt
from torch.utils.data import DataLoader
from saliency_predictor import SaliencyPredictor
from on_the_fly_augmentor import OnTheFlyAugmentor, acoustics_collate
from src.common.loss_function import (MaskedSpectrogramL1LossReduced,
                                        ExpectedKLDivergence,
                                        VecExpectedKLDivergence, 
                                        SparsityKLDivergence,
                                    )
from src.common.utils import median_mask_filtering, refining_mask_sample
from src.common.logger_SaliencyPred import SaliencyPredictorLogger
from src.common.hparams_onflyaugmentor import create_hparams
from pprint import pprint


def batchnorm_to_float(module):
    """Converts batch norm modules to FP32"""
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        batchnorm_to_float(child)
    return module


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


def plot_figures(spectrogram, posterior, mask, y, y_pred, iteration, hparams):
    # Plotting details
    pylab.xticks(fontsize=18)
    pylab.yticks(fontsize=18)
    fig, ax = pylab.subplots(3, 1, figsize=(27, 18))
    ax[0].imshow(np.log10(spectrogram + 1e-10), aspect="auto", origin="lower",
                   interpolation='none')
    ax[0].plot(151*mask, "w", linewidth=4.0)
    ax[0].set_xlabel('Time',fontsize = 20) #xlabel
    ax[0].set_ylabel('Frequency', fontsize = 20)#ylabel
    # pylab.tight_layout()
    
    ax[1].bar(np.arange(5), y, alpha=0.5, label="target")
    ax[1].bar(np.arange(5), y_pred, alpha=0.5, label="pred")
    ax[1].legend(loc=1)
    ax[1].set_xlabel('Classes',fontsize = 20) #xlabel
    ax[1].set_ylabel('Softmax Score', fontsize = 20)#ylabel
    # pylab.tight_layout()
    
    ax[2].plot(posterior, linewidth=2.5)
    ax[2].set_xlabel('Time',fontsize = 20) #xlabel
    ax[2].set_ylabel('Probability', fontsize = 20)#ylabel
    # pylab.tight_layout()
    pylab.suptitle("Utterance {}".format(iteration), fontsize=24)
    
    pylab.savefig(os.path.join(hparams.output_directory, "{}.png".format(iteration)))
    pylab.close()


def multi_sampling(model, x, y, criterion, num_samples=3):
    mask_samples = []
    posterior, mask, y_pred = model(x)
    loss = criterion(y_pred, y)
    reduced_loss = loss.item()

    y = y.squeeze().cpu().numpy()
    posterior = posterior.squeeze().detach().cpu().numpy()[:,1]
    mask = mask.squeeze().detach().cpu().numpy()[:,1]
    y_pred = y_pred.squeeze().detach().cpu().numpy()
    
    # for _ in range(11):
    #     mask = medfilt(mask, kernel_size=5)
    
    # mask_samples.append(refining_mask_sample(mask)[1])
    mask_samples.append(mask)
    
    for _ in range(num_samples-1):
        _, m, _ = model(x)
        m = m.squeeze().detach().cpu().numpy()[:,1]
        # for _ in range(11):
        #     m = medfilt(m, kernel_size=5)
        mask_samples.append(medfilt(m, kernel_size=5))
        # mask_samples.append(refining_mask_sample(m)[1])
    
    mask_intersect = np.multiply(np.logical_and(mask_samples[0], mask_samples[1]), 1)
    for i in range(2, num_samples):
        mask_intersect = np.multiply(np.logical_and(mask_intersect, mask_samples[i]), 1)
    
    for _ in range(11):
        mask_intersect = medfilt(mask_intersect, kernel_size=7)
    
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

def test(output_directory, checkpoint_path, hparams):
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

    # criterion1 = VecExpectedKLDivergence(alpha=hparams.alpha, 
                                        # beta=hparams.beta)
    criterion2 = torch.nn.L1Loss() #MSELoss
    # criterion3 = SparsityKLDivergence()

    test_loader, collate_fn = prepare_dataloaders(hparams)

    # Load checkpoint
    iteration = 0
    model, _, _, _ = load_checkpoint(checkpoint_path, model, optimizer)

    model.eval()

    chunk_array = []
    loss_array = []
    # ================ MAIN TESTING LOOP! ===================
    for i, batch in enumerate(test_loader):
        start = time.perf_counter()

        x, y, _ = batch[0].to("cuda"), batch[1].to("cuda"), batch[2]
        # input_shape should be [#batch_size, #freq_channels, #time]

        #%% Sampling masks multiple times for same utterance
        
        # x, y, y_pred, posterior, mask_sample, reduced_loss = multi_sampling(model, x, y, criterion2)
        # loss_array.append(reduced_loss)
        
        #%% Sampling the mask only once
        
        # posterior, mask_sample, y_pred = model(x)
        # loss = criterion2(y_pred, y)
        # reduced_loss = loss.item()
        # loss_array.append(reduced_loss)

        # x = x.squeeze().cpu().numpy()
        # y = y.squeeze().cpu().numpy()
        # y_pred = y_pred.squeeze().detach().cpu().numpy()
        # posterior = posterior.squeeze().detach().cpu().numpy()[:,1]
        # mask_sample = mask_sample.squeeze().detach().cpu().numpy()[:,1]
        
        # chunks, mask_sample = refining_mask_sample(mask_sample, kernel_size=7, threshold=5) # 7, 5
        # # print("\t Chunks: ", chunks)
        # chunk_array += [c[-1] for c in chunks]
        
        #%% Removing segments of length < 5
        
        posterior, mask_sample, _ = model(x)
        mask_sample = mask_sample.squeeze().detach().cpu().numpy()[:,1]
        mask_sample = random_mask_thresholding(mask_sample)
        
        mask_sample = torch.from_numpy(mask_sample).unsqueeze(dim=0).unsqueeze(dim=-1).to("cuda")
        mask_sample = mask_sample.repeat(1,1,512)
        # mask_sample = torch.zeros_like(mask_sample).to("cuda") # Zeroing out mask
        _, _, y_pred = model(x, pre_computed_mask=mask_sample)
        loss = criterion2(y_pred, y)
        reduced_loss = loss.item()
        loss_array.append(reduced_loss)

        x = x.squeeze().cpu().numpy()
        y = y.squeeze().cpu().numpy()
        y_pred = y_pred.squeeze().detach().cpu().numpy()
        posterior = posterior.squeeze().detach().cpu().numpy()[:,1]
        mask_sample = mask_sample.squeeze().detach().cpu().numpy()[:,0]
        
        chunks, mask_sample = refining_mask_sample(mask_sample, kernel_size=7, threshold=5) # 7, 5
        # print("\t Chunks: ", chunks)
        chunk_array += [c[-1] for c in chunks]
        
        #%% Plotting
        plot_figures(x, posterior, mask_sample, y, y_pred, iteration+1, hparams)

        if not math.isnan(reduced_loss):
            duration = time.perf_counter() - start
            print("Test loss {} {:.6f} {:.2f}s/it".format(
                iteration, reduced_loss, duration))

        iteration += 1
    
    print("Avg. Loss: {:.3f}".format(np.mean(loss_array)))
    
    return chunk_array


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
                                        "images_check"
                                    )

    if not hparams.output_directory:
        raise FileExistsError('Please specify the output dir.')
    else:
        if not os.path.exists(hparams.output_directory):
            os.makedirs(hparams.output_directory)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    chunk_array = test(
                        hparams.output_directory,
                        hparams.checkpoint_path,
                        hparams,
                    )
