#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 16:55:19 2022

@author: ravi
"""

import sys
import torch
import logging
import joblib
import random
import numpy as np
import scipy.io.wavfile as scwav
import speechbrain as sb
from tqdm.contrib import tqdm
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

from src.common.hparams import create_hparams
from src.common.utils import load_filepaths
# from src.common.utterance import Utterance

logger = logging.getLogger(__name__)

class PPGASRLoader(sb.Brain):
    """Class that manages the training loop. See speechbrain.core.Brain."""
    
    def __init__(self, yaml_params, data_utterance_paths, hparams):
        
        # Speechbrain model loading and directory settings
        hparams_file, run_opts, overrides = sb.parse_arguments([yaml_params])
        
        with open(hparams_file) as fin:
            self.hparams_sb = load_hyperpyyaml(fin, overrides)
        
        sb.create_experiment_directory(
            experiment_directory=self.hparams_sb["output_folder"],
            hyperparams_to_save=hparams_file,
            overrides=overrides,
        )
        
        run_on_main(self.hparams_sb["pretrainer"].collect_files)
        self.hparams_sb["pretrainer"].load_collected(device=run_opts["device"])
        
        # Initializing the speechbrain ASR model
        super(PPGASRLoader, self).__init__(
            modules=self.hparams_sb["modules"],
            opt_class=self.hparams_sb["opt_class"],
            hparams=self.hparams_sb,
            run_opts=run_opts,
            checkpointer=self.hparams_sb["checkpointer"],
        )
        
        # Setting ASR model to run in inference mode
        self.on_evaluate_start(max_key=None, min_key=None)
        self.on_stage_start(sb.Stage.TEST, epoch=None)
        self.modules.eval()
        
        # PPG extraction and PPG2MEL model hyperparams
        self.data_utterance_paths = load_filepaths(data_utterance_paths)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.is_full_ppg = hparams.is_full_ppg
        self.is_append_f0 = hparams.is_append_f0
        self.is_cache_feats = hparams.is_cache_feats
        self.load_feats_from_disk = hparams.load_feats_from_disk
        self.feats_cache_path = hparams.feats_cache_path
        self.ppg_subsampling_factor = hparams.ppg_subsampling_factor

        if self.is_cache_feats and self.load_feats_from_disk:
            raise ValueError('If you are loading feats from the disk, do not '
                             'rewrite them back!')

        # self.stft = layers.TacotronSTFT(
        #     hparams.filter_length, hparams.hop_length, hparams.win_length,
        #     hparams.n_acoustic_feat_dims, hparams.sampling_rate,
        #     hparams.mel_fmin, hparams.mel_fmax)
        random.seed(hparams.seed)
        random.shuffle(self.data_utterance_paths)

        self.ppg_sequences = []
        self.acoustic_sequences = []
        if self.load_feats_from_disk:
            print('Loading data from %s.' % self.feats_cache_path)
            with open(self.feats_cache_path, 'rb') as f:
                data = joblib.load(f)
            self.ppg_sequences = data[0]
            self.acoustic_sequences = data[1]
        else:
            for utterance_path in tqdm(self.data_utterance_paths):
                ppg_feat_pair = self.extract_utterance_feats(utterance_path)
                # print("PPG dimensions: ", ppg_feat_pair[0].shape)
                # print("STFT dimensions: ", ppg_feat_pair[1].shape)
                self.ppg_sequences.append(ppg_feat_pair[0].astype(
                    np.float32))
                self.acoustic_sequences.append(ppg_feat_pair[1])
        if self.is_cache_feats:
            print('Caching data to %s.' % self.feats_cache_path)
            with open(self.feats_cache_path, 'wb') as f:
                joblib.dump([self.ppg_sequences, self.acoustic_sequences], f)

    def extract_utterance_feats(self, data_utterance_path, is_full_ppg=False):
        """Get PPG and Mel (+ optional F0) for an utterance.

        Args:
            data_utterance_path: The path to the data utterance protocol buffer.
            is_full_ppg: If True, will use the full PPGs.

        Returns:
            feat_pairs: A list, each is a [ppg, mel] pair.
        """
        # utt = Utterance()
        fs, wav = scwav.read(data_utterance_path)
        # utt.fs = fs
        # utt.wav = wav
        wav = torch.from_numpy(np.expand_dims(wav, axis=0)).float().to(self.device)
        wavlen = torch.Tensor([1]).float().to(self.device)

        stft_feats = self.hparams.compute_features(wav)
        stft_feats = self.modules.normalize(stft_feats, wavlen)
        ppg = self.modules.encoder(stft_feats)
        
        # stft_feats = stft_feats.squeeze(axis=0).cpu().numpy()
        # (
        #     stft_mean,
        #     stft_std,
        # ) = (np.mean(stft_feats, axis=0), np.std(stft_feats, axis=0))
        # stft_feats = (stft_feats - stft_mean) / (1e-10 + stft_std)
        # stft_feats = torch.from_numpy(stft_feats).float().to(self.device)
        
        # ppg_feats = ppg.squeeze(axis=0).cpu().detach().numpy()
        # (
        #     ppg_mean,
        #     ppg_std,
        # ) = (np.mean(ppg_feats, axis=0), np.std(ppg_feats, axis=0))
        # ppg_feats = (ppg_feats - ppg_mean) / (1e-10 + ppg_std)
        
        stft_feats = stft_feats.squeeze(axis=0).cpu().numpy()
        (
            stft_mean,
            stft_std,
        ) = (np.mean(stft_feats), np.std(stft_feats))
        stft_feats = (stft_feats - stft_mean) / (1e-10 + stft_std)
        stft_feats = torch.from_numpy(stft_feats).float().to(self.device)
        
        ppg_feats = ppg.squeeze(axis=0).cpu().detach().numpy()
        (
            ppg_mean,
            ppg_std,
        ) = (np.mean(ppg_feats), np.std(ppg_feats))
        ppg_feats = (ppg_feats - ppg_mean) / (1e-10 + ppg_std)

        return [ppg_feats, stft_feats]

    def __getitem__(self, index):
        """Get a new data sample in torch.float32 format.

        Args:
            index: An int.

        Returns:
            T*D1 PPG sequence, T*D2 mels
        """
        if self.ppg_subsampling_factor == 1:
            curr_ppg = self.ppg_sequences[index]
        else:
            curr_ppg = self.ppg_sequences[index][
                        0::self.ppg_subsampling_factor, :]

        return torch.from_numpy(curr_ppg), self.acoustic_sequences[index]

    def __len__(self):
        return len(self.ppg_sequences)

def ppg_acoustics_collate(batch):
    """Zero-pad the PPG and acoustic sequences in a mini-batch.

    Also creates the stop token mini-batch.

    Args:
        batch: An array with B elements, each is a tuple (PPG, acoustic).
        Consider this is the return value of [val for val in dataset], where
        dataset is an instance of PPGSpeechLoader.

    Returns:
        ppg_padded: A (batch_size, feature_dim_1, num_frames_1) tensor.
        input_lengths: A batch_size array, each containing the actual length
        of the input sequence.
        acoustic_padded: A (batch_size, feature_dim_2, num_frames_2) tensor.
        gate_padded: A (batch_size, num_frames_2) tensor. If "1" means reaching
        stop token. Currently assign "1" at the last frame and the padding.
        output_lengths: A batch_size array, each containing the actual length
        of the output sequence.
    """
    # Right zero-pad all PPG sequences to max input length.
    # x is (PPG, acoustic), x[0] is PPG, which is an (L(varied), D) tensor.
    input_lengths, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([x[0].shape[0] for x in batch]), dim=0,
        descending=True)
    max_input_len = input_lengths[0]
    ppg_dim = batch[0][0].shape[1]

    ppg_padded = torch.FloatTensor(len(batch), max_input_len, ppg_dim)
    ppg_padded.zero_()
    for i in range(len(ids_sorted_decreasing)):
        curr_ppg = batch[ids_sorted_decreasing[i]][0]
        ppg_padded[i, :curr_ppg.shape[0], :] = curr_ppg

    # Right zero-pad acoustic features.
    feat_dim = batch[0][1].shape[1]
    max_target_len = max([x[1].shape[0] for x in batch])
    # Create acoustic padded and gate padded
    acoustic_padded = torch.FloatTensor(len(batch), max_target_len, feat_dim)
    acoustic_padded.zero_()
    gate_padded = torch.FloatTensor(len(batch), max_target_len)
    gate_padded.zero_()
    output_lengths = torch.LongTensor(len(batch))
    for i in range(len(ids_sorted_decreasing)):
        curr_acoustic = batch[ids_sorted_decreasing[i]][1]
        acoustic_padded[i, :curr_acoustic.shape[0], :] = curr_acoustic
        gate_padded[i, curr_acoustic.shape[0] - 1:] = 1
        output_lengths[i] = curr_acoustic.shape[0]

    ppg_padded = ppg_padded.transpose(1, 2)
    acoustic_padded = acoustic_padded.transpose(1, 2)

    return ppg_padded, input_lengths, acoustic_padded, gate_padded,\
        output_lengths


# def utt_to_sequence(utt: Utterance):
#     """Get PPG tensor for inference.

#     Args:
#         utt: A data utterance object.
#         is_full_ppg: If True, will use the full PPGs.
#         is_append_f0: If True, will append F0 features.

#     Returns:
#         A 1*D*T tensor.
#     """
#     ppg = utt.ppg

#     return torch.from_numpy(ppg).float().transpose(0, 1).unsqueeze(0)

if __name__ == "__main__":
    hparams = create_hparams()

    ppg = PPGASRLoader(
            yaml_params="/home/ravi/Desktop/fac-via-ppg/acoustic_modeling/train.yaml",
            data_utterance_paths=hparams.training_files, 
            hparams=hparams,
        )
    
    # from speechbrain.pretrained import EncoderDecoderASR

    # asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="pretrained_model")
    # # asr_model = EncoderDecoderASR.from_hparams(source="./speechbrain_results/CRDNN_BPE_960h_LM/2603", savedir="save")
    # audio_file = '/home/ravi/Downloads/JHU_Emotional/speech_data_male1/neutral/50.wav'
    # print(asr_model.transcribe_file(audio_file))































