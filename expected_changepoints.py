#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 17:12:10 2023

@author: ravi
"""


import numpy as np
import scipy.stats as scistat

#%%
lengths = [np.random.randint(50, 300) for _ in range(1000)]

(alpha, beta) = (0.1, 0.95)
data = []
for l in lengths:
    seq = [scistat.bernoulli.rvs(alpha)]
    for i in range(1, l):
        if seq[-1] == 1:
            seq.append(scistat.bernoulli.rvs(beta))
        else:
            seq.append(scistat.bernoulli.rvs(1 - beta))
    data.append(seq)
    
count_changepoints = []
for d in data:
    cc = 0
    for l in range(len(d)-1):
        if d[l] != d[l+1]:
            cc += 1
    count_changepoints.append(cc)

print("average changepoints: ", np.mean(count_changepoints))


#%%
alpha = 0.1
beta = 0.99
factor = 0.993 #0.942915

lengths = [np.random.randint(50, 300) for _ in range(1000)]

data = []
beta_data = []
for l in lengths:
    beta_data.append([beta])
    seq = [scistat.bernoulli.rvs(alpha)]
    for i in range(1, l):
        if seq[-1] == 1:
            new_sample = scistat.bernoulli.rvs(beta)
            if new_sample == seq[-1]:
                beta = beta * factor
            else:
                beta = 0.9
            seq.append(new_sample)
        else:
            new_sample = scistat.bernoulli.rvs(1 - beta)
            if new_sample == seq[-1]:
                beta = beta * factor
            else:
                beta = 0.9
            seq.append(new_sample)
        beta_data[-1].append(beta)
    data.append(seq)
    
count_changepoints = []
for d in data:
    cc = 0
    for l in range(len(d)-1):
        if d[l] != d[l+1]:
            cc += 1
    count_changepoints.append(cc)

print("average changepoints: ", np.mean(count_changepoints))