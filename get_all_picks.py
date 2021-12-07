# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 23:25:40 2021

@author: xuwc
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 19:15:26 2021

@author: xuwc
"""

import numpy as np
from obspy import read
from scipy import signal
from scipy.signal import find_peaks

def butter_bandpass_filtfilt(data,cutoff,fs,order=4):
    if data.ndim == 2:
        axis = 1
    else: 
        axis = -1
    wn = 2*cutoff/fs
    b, a = signal.butter(order, wn, 'bandpass', analog = False)
    output = signal.filtfilt(b, a, data, axis=axis)
    return output
def get_std(x,data):
    data1 = data[250:0:-1]
    data2 = data[-251:-1]
    data = np.hstack((data1,data,data2))
    tmp = np.zeros([len(x),500])
    for i in range(len(x)):
        tmp[i,:] = data[x[i]:x[i]+500]
    return np.std(tmp,axis=1)
def get_peaks(data):
    h = 300
    d = 200
    data = butter_bandpass_filtfilt(data, np.array([0.001,2]), 20,2)
    
    data = data.astype(int)
    ###########################################
    data2 = data.copy()
    data2[data2<0] = 0
    # indexes2 = peakutils.indexes(data2,thres=100,min_dist=100,thres_abs=True)
    indexes2 = find_peaks(data2,height=h,distance=d)[0]
    data1 = -data
    data1[data1<0] = 0
    # indexes1 = peakutils.indexes(data1,thres=100,min_dist=100,thres_abs=True)
    indexes1 = find_peaks(data1,height=h,distance=d)[0]
    
    indexes = np.sort(np.hstack((indexes1,indexes2)))
    amp = np.abs(data[indexes])
    std = get_std(indexes,data)
    k = 2
    id2 = indexes[(amp-k*std)>0]
    amp = np.abs(data[id2])
    # remove close peaks
    if len(id2) <= 2:
        return (id2,data[id2])
    i = 2
    while 1:
        if id2[i]-id2[i-1] < 200:
            if amp[i] > amp[i-1]:
                amp = np.delete(amp,i-1)
                id2 = np.delete(id2,i-1)
            else:
                amp = np.delete(amp,i)
                id2 = np.delete(id2,i)
            i = i-1
        i = i+1
        if i == len(id2):
            break
    return (id2,data[id2])