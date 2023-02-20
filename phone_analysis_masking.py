#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 13:19:35 2023

@author: ravi
"""


import json
import joblib
import os


chunk_phone_data = joblib.load("./masked_predictor_output/libri_1e-05_10.0_2e-07_5.0_tflayers_3_batchsize_8/test_chunks_array.pkl")
with open("./grouped_consonants_vowels.json", "rb") as f:
    phone_classes = json.load(f)

consonants = phone_classes["consonants"] # nasals, plosives, fricatives, approximants
vowels = phone_classes["vowels"] # stress_0, stress_1, stress_2

