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
import textgrid

from torch.utils.data import DataLoader
from src.common.hparams_onflyaugmentor import create_hparams
from on_the_fly_augmentor import OnTheFlyAugmentor, acoustics_collate


def prepare_dialog_lookup():
    vk = pd.read_csv("/home/ravi/Downloads/VESUS-Emotion-Recog/VESUS_Key.csv")
    file_dialog = vk[["File Path", "Actor Dialog"]]

    file_dialog_dict = file_dialog.set_index("File Path").T.to_dict("list")
    return file_dialog_dict


def cleanup_text(text):
    chars2int = [ord(i) for i in text]
    for i, c in enumerate(chars2int):
        if c == 8217:
            chars2int[i] = 39
    
    int2chars = [chr(i) for i in chars2int]
    return ("").join(int2chars)


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


def get_textgrid(audio_path, text_path, textgrid_path):
    os.system("python2 /home/ravi/Penn_fa/aligner/align.py {0} {1} {2}".format(
                                                            audio_path,
                                                            text_path,
                                                            textgrid_path,
                                                            ))
    tgd = textgrid.TextGrid.fromFile(textgrid_path)
    return tgd


def format_audio_text(data_object, index, lookup_dict):
    utterance_path = data_object.utterance_rating_paths[index].split(" ,")[0]
    text_key = data_object.utterance_rating_paths[index].split(" ,")[0].split("/")[-3:]
    lookup_key = "/" + ("/").join(text_key)
    text = lookup_dict[lookup_key][0]
    text = cleanup_text(text)
    
    temp_audio_loc = "/home/ravi/Desktop/fac-ppg/temp/audio.wav"
    temp_text_loc = "/home/ravi/Desktop/fac-ppg/temp/text.txt"
    temp_grid_loc = "/home/ravi/Desktop/fac-ppg/temp/grid.textgrid"
    
    data, sr = sf.read(utterance_path)
    data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    new_data = (data * 32767).astype(np.int16)
    scwav.write(temp_audio_loc, 16000, new_data)

    with open(temp_txt_loc, "w") as f:
        f.writelines(text)
        f.close()
    
    txt_grid = get_textgrid(audio_path=temp_audio_loc, 
                            text_path=temp_text_loc, 
                            textgrid_path=temp_grid_loc,
                            )
    
    return txt_grid


if __name__ == "__main__":
    hparams = create_hparams()
    data_object, _, _ = prepare_dataloaders(hparams)
    lookup_dict = prepare_dialog_lookup()
    
    temp_dir = "/home/ravi/Desktop/fac-ppg/temp"
    temp_audio_loc = os.path.join(temp_dir, "audiofile.wav")
    temp_txt_loc = os.path.join(temp_dir, "textfile.txt")
    # temp_grd_loc = os.path.join(temp_dir, "textgridfile.textgrid")
    
    for i in range(len(data_object)):
        temp_grd_loc = os.path.join(temp_dir, "{}_textgridfile.textgrid".format(i))
        utterance_path = data_object.utterance_rating_paths[i].split(" ,")[0]
        data, sr = sf.read(utterance_path)
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        new_data = (data * 32767).astype(np.int16)
        scwav.write(temp_audio_loc, 16000, new_data)

        x = data_object.utterance_rating_paths[i].split(" ,")[0].split("/")[-3:]
        lookup_key = "/" + ("/").join(x)
        text = lookup_dict[lookup_key][0]
        text = cleanup_text(text)

        with open(temp_txt_loc, "w") as f:
            f.writelines(text)
            f.close()
        
        txt_grid = get_textgrid(audio_path=temp_audio_loc, 
                                text_path=temp_txt_loc, 
                                textgrid_path=temp_grd_loc,
                                )

        # os.system("python2 /home/ravi/Penn_fa/aligner/align.py {0} {1} {2}".format(temp_audio_loc, 
        #                                                                            temp_txt_loc, 
        #                                                                            temp_grd_loc))
        break
    














