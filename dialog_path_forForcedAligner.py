#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 16:09:01 2023

@author: ravi
"""


import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from src.common.hparams_onflyaugmentor import create_hparams
from on_the_fly_augmentor import OnTheFlyAugmentor, acoustics_collate


def prepare_dialog_lookup():
    vk = pd.read_csv("/home/ravi/Downloads/VESUS-Emotion-Recog/VESUS_Key.csv")
    file_dialog = vk[["File Path", "Actor Dialog"]]

    file_dialog_dict = file_dialog.set_index("File Path").T.to_dict("list")
    return file_dialog_dict


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
    return testset, test_loader, collate_fn


if __name__ == "__main__":
    hparams = create_hparams()
    data_object, dataloader, _ = prepare_dataloaders(hparams)
    
    x = data_object.utterance_rating_paths[0].split(" ,")[0].split("/")[-3:]
    lookup_key = "/" + ("/").join(x)
    
    
    
