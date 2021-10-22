#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Mining final project: Ensemble

@author: Jui-Hsiu, Hsu
"""
import numpy as np
import pandas as pd

# 1. Read Data
LR = pd.read_csv("submit/LR.csv")
DNN = pd.read_csv("submit/DNN.csv")
SVR = pd.read_csv("submit/SVR.csv")
RFR = pd.read_csv("submit/RFR.csv")

# 2. Ensemble
average_votes = (DNN['Votes'] + SVR['Votes'] + RFR['Votes']) / 3
result = pd.concat([LR['Id'], average_votes], axis=1)
result.to_csv('submit/ensemble.csv', index=False)