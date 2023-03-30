#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 13:19:35 2023

@author: ravi
"""

import json
import joblib
import os
import numpy as np
import pylab
import itertools

#%% Functions

index2emotion = {
                    0:"neutral",
                    1:"angry",
                    2:"happy",
                    3:"sad",
                    4:"fearful",
                }

def get_phone_class(phone, monophone_dict):
    keys = ["nasals", "plosives", "fricatives", "approximants"]

    for k in keys:
        if phone.upper() in monophone_dict["consonants"][k]:
            return k

    if phone.upper() in (
                        monophone_dict["vowels"]["stress_0"]
                        + monophone_dict["vowels"]["stress_1"]
                        + monophone_dict["vowels"]["stress_2"]
                    ):
        return "vowels"
    else:
        return "sil"


def get_target_emotions(target_saliency):
    max_val = np.max(target_saliency)
    indices = [i for i in range(5) if target_saliency[i]==max_val]
    return [index2emotion[i] for i in indices]

#%% Read the important data

chunk_phone_data = joblib.load("./masked_predictor_output/1e-05_10.0_0.0002_5.0_BiLSTM_maskGen/test_chunks_array.pkl")
with open("./grouped_consonants_vowels.json", "rb") as f:
    monophone_dict = json.load(f)
    f.close()

chunks_array = chunk_phone_data["chunks"]
targets_array = chunk_phone_data["targets"]
phones_array = chunk_phone_data["phones"]
del chunk_phone_data, f

#%% Get all consonants and vowels

consonants = monophone_dict["consonants"] # nasals, plosives, fricatives, approximants
vowels = monophone_dict["vowels"] # stress_0, stress_1, stress_2

all_vowels = (
                vowels["stress_0"]
                + vowels["stress_1"]
                + vowels["stress_2"]
            )

all_consonants = (
                    consonants["nasals"] 
                  + consonants["plosives"] 
                  + consonants["fricatives"] 
                  + consonants["approximants"]
                  )

monophone_wt = {"nasals": 3,
              "plosives": 6,
              "fricatives": 11,
              "approximants": 4,
              "vowels":45,
              "sil":2}

biphone_wt = {}
for c1 in list(monophone_wt.keys()):
    for c2 in list(monophone_wt.keys()):
        biphone_wt[c1+"_"+c2] = monophone_wt[c1]*monophone_wt[c2]

del c1, c2

#%% Create class-pair dictionary

phone_classes = ["nasals", "plosives", "fricatives", "approximants", "vowels", "sil"]
pairwise_dicts = {}
mono_dicts = {}
biphone_wt = {}

for pair in itertools.combinations(phone_classes, 2):
    pairwise_dicts[pair[0]+"_"+pair[1]] = 0
    biphone_wt[pair[0]+"_"+pair[1]] = monophone_wt[pair[0]]*monophone_wt[pair[1]]

for mono in phone_classes:
    mono_dicts[mono] = 0
    pairwise_dicts[mono+"_"+mono] = 0
    biphone_wt[mono+"_"+mono] = monophone_wt[mono]*monophone_wt[mono]

# for c1 in phone_classes:
#     mono_dicts[c1] = 0
#     for c2 in phone_classes:
#         pairwise_dicts[c1+"_"+c2] = 0
# del c1, c2
del pair, mono

#%% Emotion wise analysis

emotion_classes = ["neutral", "angry", "happy", "sad", "fearful"]
emotional_data = {}

for e in emotion_classes:
    emotional_data[e] = {"biclass": pairwise_dicts.copy(), 
                         "monoclass": mono_dicts.copy()}
del e

#%% Monophones

for (chunk_mask, target, phones) in zip(chunks_array, targets_array, phones_array):
    emotions = get_target_emotions(target)
    chunks, mask = chunk_mask[0], chunk_mask[1]
    for p in phones:
        if np.sum(mask[p[3]:p[4]]) == (p[4] - p[3]):
            phone_class = get_phone_class(p[0], monophone_dict)
            for e in emotions:
                emotional_data[e]["monoclass"][phone_class] += 1/monophone_wt[phone_class]

pylab.xticks(fontsize=5)
pylab.yticks(fontsize=5)
fig, ax = pylab.subplots(1, 5, figsize=(35, 15))
# neutral_data = np.asarray(list(emotional_data["neutral"]["monoclass"].values()))
for i in range(5):
    emotion = emotion_classes[i]
    labels = list(emotional_data[emotion]["monoclass"].keys())
    data = np.asarray(list(emotional_data[emotion]["monoclass"].values()))
    # data = data / neutral_data
    data /= np.sum(data)
    ax[i].bar(labels, data, label=emotion)
    ax[i].legend(loc=1), ax[i].title.set_text(emotion)
    ax[i].set_xticklabels(labels=labels, fontsize=13, rotation=90, fontweight="bold")
    # pylab.tight_layout()

pylab.suptitle("Monophones")
pylab.savefig("/home/ravi/Desktop/monophones_saliency.png")
pylab.close("all")

#%% Phone boundaries

for (chunk_mask, target, phones) in zip(chunks_array, targets_array, phones_array):
    emotions = get_target_emotions(target)
    chunks, mask = chunk_mask[0], chunk_mask[1]
    for n in range(len(phones)-1):
        p_left = phones[n]
        p_right = phones[n+1]
        index = (p_left[4] + p_right[3])//2

        if mask[index]>0:
            phone_class_l = get_phone_class(p_left[0], monophone_dict)
            phone_class_r = get_phone_class(p_right[0], monophone_dict)
            if phone_class_l+"_"+phone_class_r in pairwise_dicts.keys():
                key = phone_class_l+"_"+phone_class_r
            else:
                key = phone_class_r+"_"+phone_class_l
            for e in emotions:
                emotional_data[e]["biclass"][key] += 1/biphone_wt[key]


pylab.xticks(fontsize=5)
pylab.yticks(fontsize=5)
fig, ax = pylab.subplots(1, 5, figsize=(40, 20))
# neutral_data = np.asarray(list(emotional_data["neutral"]["biclass"].values())) + 1e-6
for i in range(5):
    emotion = emotion_classes[i]
    labels = list(emotional_data[emotion]["biclass"].keys())
    data = np.asarray(list(emotional_data[emotion]["biclass"].values())) + 1e-6
    # data = data / neutral_data
    data /= np.sum(data)
    ax[i].bar(labels, data, label=emotion)
    ax[i].legend(loc=1), ax[i].title.set_text(emotion)
    ax[i].set_xticklabels(labels=labels, fontsize=10, rotation=90, fontweight="bold")
    # pylab.tight_layout()

pylab.suptitle("Biphones (boundaries)")
pylab.savefig("/home/ravi/Desktop/biphones_saliency.png")
pylab.close("all")

#%% Relative Monophones

for (chunk_mask, target, phones) in zip(chunks_array, targets_array, phones_array):
    emotions = get_target_emotions(target)
    chunks, mask = chunk_mask[0], chunk_mask[1]
    for p in phones:
        if np.sum(mask[p[3]:p[4]]) == (p[4] - p[3]):
            phone_class = get_phone_class(p[0], monophone_dict)
            for e in emotions:
                emotional_data[e]["monoclass"][phone_class] += 1/monophone_wt[phone_class]

pylab.xticks(fontsize=5)
pylab.yticks(fontsize=5)
fig, ax = pylab.subplots(1, 4, figsize=(35, 15))
neutral_data = np.asarray(list(emotional_data["neutral"]["monoclass"].values()))
for i in range(1, 5):
    emotion = emotion_classes[i]
    labels = list(emotional_data[emotion]["monoclass"].keys())
    data = np.asarray(list(emotional_data[emotion]["monoclass"].values()))
    data = data / neutral_data
    data /= np.sum(data)
    ax[i-1].bar(labels, data, label=emotion)
    ax[i-1].legend(loc=1)
    ax[i-1].title.set_text(emotion)
    ax[i-1].set_xticklabels(labels=labels, fontsize=13, rotation=90, fontweight="bold")
    # pylab.tight_layout()

pylab.suptitle("Relative Monophones")
pylab.savefig("/home/ravi/Desktop/monophones_saliency_relative.png")
pylab.close("all")

#%% Relative Phone boundaries

for (chunk_mask, target, phones) in zip(chunks_array, targets_array, phones_array):
    emotions = get_target_emotions(target)
    chunks, mask = chunk_mask[0], chunk_mask[1]
    for n in range(len(phones)-1):
        p_left = phones[n]
        p_right = phones[n+1]
        index = (p_left[4] + p_right[3])//2

        if mask[index]>0:
            phone_class_l = get_phone_class(p_left[0], monophone_dict)
            phone_class_r = get_phone_class(p_right[0], monophone_dict)
            if phone_class_l+"_"+phone_class_r in pairwise_dicts.keys():
                key = phone_class_l+"_"+phone_class_r
            else:
                key = phone_class_r+"_"+phone_class_l
            for e in emotions:
                emotional_data[e]["biclass"][key] += 1/biphone_wt[key]


pylab.xticks(fontsize=5)
pylab.yticks(fontsize=5)
fig, ax = pylab.subplots(1, 4, figsize=(40, 20))
neutral_data = np.asarray(list(emotional_data["neutral"]["biclass"].values())) + 1e-6
for i in range(1, 5):
    emotion = emotion_classes[i]
    labels = list(emotional_data[emotion]["biclass"].keys())
    data = np.asarray(list(emotional_data[emotion]["biclass"].values())) + 1e-6
    data = data / neutral_data
    data /= np.sum(data)
    ax[i-1].bar(labels, data, label=emotion)
    ax[i-1].legend(loc=1)
    ax[i-1].title.set_text(emotion)
    ax[i-1].set_xticklabels(labels=labels, fontsize=10, rotation=90, fontweight="bold")
    # pylab.tight_layout()

pylab.suptitle("Relative Biphones (boundaries)")
pylab.savefig("/home/ravi/Desktop/biphones_saliency_relative.png")
pylab.close("all")


























































