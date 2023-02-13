#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 16:09:01 2023

@author: ravi
"""

import os
import librosa

import pandas as pd
import numpy as np
import soundfile as sf
import scipy.io.wavfile as scwav

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
    data_object, _, _ = prepare_dataloaders(hparams)
    lookup_dict = prepare_dialog_lookup()
    
    x = data_object.utterance_rating_paths[0].split(" ,")[0].split("/")[-3:]
    lookup_key = "/" + ("/").join(x)
    text = lookup_dict[lookup_key][0]

    with open("/home/ravi/tmp/textfile.txt", "w") as f:
        f.writelines(text)
        f.close()
    
    data, sr = sf.read(data_object.utterance_rating_paths[0].split(" ,")[0])
    data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    scwav.write("/home/ravi/tmp/audiofile.wav", 16000, np.asarray(data, np.int16))

    os.system("python2 /home/ravi/Penn_fa/aligner/align.py /home/ravi/tmp/audiofile.wav /home/ravi/tmp/textfile.txt /home/ravi/tmp/textgridfile.textgrid")
    
