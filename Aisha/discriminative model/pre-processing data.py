#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 00:48:45 2020

@author: xiaoyanchen
"""

import numpy as np
from data_loading import real_data_loading
ori_data = real_data_loading('stock',24)
n = int(len(ori_data)/2)
ori_data_1 = ori_data[:n]
ori_data_2 = ori_data[n:]
with open('stock3ddata.npy', 'wb') as f:
    np.save(f, np.array(ori_data_1))
    np.save(f,np.array(ori_data_2))