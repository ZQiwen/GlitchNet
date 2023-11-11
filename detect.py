# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 23:30:48 2021

@author: xuwc
"""
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from obspy import read,Stream
from get_all_picks import get_peaks
import os
import io
import time
import warnings
from paras import INPUT_FOLDER,MODEL,IF_PLOT,SAVE_GLITCH_CATALOG,SP_DATA

def read_large_mseed(filenm):
    st0 = Stream()
    reclen = 512
    chunksize = 1000000 * reclen # Around 500 MB
    
    with io.open(filenm, "rb") as fh:
        while True:
            with io.BytesIO() as buf:
                c = fh.read(chunksize)
                if not c:
                    break
                buf.write(c)
                buf.seek(0, 0)
                st = read(buf)
            # Do something useful!
            print(st)
            st0 += st
    return st0

def normalization(x):
    x = x-np.mean(x)
    return x/np.std(x)
def normalization_array(x):
    x = x-x.mean(axis=1,keepdims=True)
    return x/x.std(axis=1,keepdims=True)
def load_detection_model(modelnm):
    model = tf.keras.models.load_model(modelnm)
    return model
def cut_data(data,idx):
    n = len(idx)
    dat = np.zeros([n,600])
    for i in range(n):
        dat[i,:] = data[idx[i]-160:idx[i]+440]
    return dat
def cut_data_sp(data,idx):
    n = len(idx)
    dat = np.zeros([n,1200])
    for i in range(n):
        dat[i,:] = data[idx[i]-195:idx[i]+1005]
    return dat
def prediction(dat,model):
    dat = np.reshape(dat,(dat.shape[0],dat.shape[1],1))
    pre = model.predict(dat)   
    pre = np.reshape(pre,[len(pre),])
    return pre
def write_starttime(starttime):
    return str(starttime.year)+str(starttime.month).zfill(2)+str(starttime.day).zfill(2)+str(starttime.hour).zfill(2)+str(starttime.minute).zfill(2)+str(starttime.second).zfill(2)
t1 = time.time()
modelnm = MODEL
# modelnm = 'C:/Users/xwccc/Desktop/bestmodel/apmodels/test_4ksize_4lr_0.001time_20210402-195547'
model = load_detection_model(modelnm)
# model.summary()
# 
filelist = os.listdir(INPUT_FOLDER)
for filenm in filelist:
    file = INPUT_FOLDER+'/'+filenm
    print('processing '+file)
    st = read_large_mseed(file)
    dataset = [trace.data for trace in st]
    starttime = [trace.stats.starttime for trace in st]
    for tnum in range(len(dataset)):
        data = dataset[tnum]
        if len(data) < 600 :
            warnings.warn('Trace too short, no detection processes are implemented :' + st[tnum].__str__())
            continue
        idx,amp = get_peaks(data) 
        if not SP_DATA:
            idx = idx[idx > 161]
            idx = idx[idx < len(data)-441]
            dat = cut_data(data, idx)
        else:
            idx = idx[idx > 196]
            idx = idx[idx < len(data)-1006]
            dat = cut_data_sp(data, idx)
        if len(idx) == 0:
            warnings.warn('No glitch found :' + st[tnum].__str__())
            continue
        dat = normalization_array(dat)
        pre = prediction(dat, model)
        idx = idx[pre>0.8]
        if len(idx) == 0:
            warnings.warn('No glitch found :' + st[tnum].__str__())
            continue
        np.save('catalog_for_removal/'+filenm[:-6]+st[tnum].stats.channel+write_starttime(st[tnum].stats.starttime)+'.npy',idx)
        if IF_PLOT:
            plt.figure(figsize=[15,7])
            if not SP_DATA:
                plt.plot(np.arange(0,len(data)/20,1/20),data)
                plt.scatter(idx/20,data[idx],c='r',s=30)
                plt.xlim([0,len(data)/20])
                plt.ylim([np.min(data)-300,np.max(data)+300])
                plt.title('starttime = '+str(starttime[tnum]))                
        if SAVE_GLITCH_CATALOG:
            if len(idx) > 0:
                with open('catalog/'+filenm[:-6]+st[tnum].stats.channel+write_starttime(st[tnum].stats.starttime)+'.txt','w') as f:
                    starttime = st[tnum].stats.starttime
                    peak_time = [starttime+dt/20 for dt in idx]
                    f.writelines([t.__str__()+'\n' for t in peak_time])
t2 = time.time()
print(t2-t1)

