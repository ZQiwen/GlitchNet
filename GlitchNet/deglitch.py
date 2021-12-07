# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 11:03:56 2021

@author: Chiween
"""

from au import AutoEncoder
from obspy import read
import numpy as np
import matplotlib.pyplot as plt
import peakutils
from time import time 
from paras import *
import os
from new_detect import read_large_mseed

# ==================================================
# %% load autoencoder from file 
# ==================================================

autoencoder = AutoEncoder(npt, [256, 128, 64, 32], [['rbmw1', 'rbmhb1', 'rbmha1'],
                                                     ['rbmw2', 'rbmhb2', 'rbmha2'],
                                                     ['rbmw3', 'rbmhb3', 'rbmha3'],
                                                     ['rbmw4', 'rbmhb4', 'rbmha4']], 
                          tied_weights=False)

autoencoder.load_weights('./out-VBB/au.chp')

# =============================================================================================
# %% ================= CASE 2 REMOVAL of GLITCHES for DAILY SEISMOGRAMS ======================
# =============================================================================================
def write_starttime(starttime):
    return str(starttime.year)+str(starttime.month).zfill(2)+str(starttime.day).zfill(2)+str(starttime.hour).zfill(2)+str(starttime.minute).zfill(2)+str(starttime.second).zfill(2)

data_dir=r'catalog_for_removal'

t1=time()

mseed_files=os.listdir(INPUT_FOLDER)
if not os.path.isdir('Output'):
  os.mkdir('Output')


for filename in mseed_files:
  
  st = read_large_mseed(INPUT_FOLDER+'/%s' % filename)
  print('processing '+ filename)
  for i_component in range(len(st)):
    # ------------------------------
    # ---------- data --------------
    # ------------------------------
    
    raw_obs = st[i_component].data.astype(np.float32)
    
    npt_raw_obs = raw_obs.shape[0]
    assert npt_raw_obs > npt
    
    # date=filename.split('.')[0].split('_')[-1]
    
    glitch_list = np.load(data_dir+'/'+filename[:-6]+st[i_component].stats.channel+write_starttime(st[i_component].stats.starttime)+'.npy')
    glitch_list = glitch_list.astype(int) - 160    
    n_glitch=glitch_list.shape[0]
    assert n_glitch > 0
    
    # ------------------------------
    # ---------- deglitch ----------
    # ------------------------------
    
    
    if PLOT_FIGURE:
      tshift_plot=glitch_list.copy()
      reconstructed_glitch = np.array(
            [raw_obs[t+np.arange(npt)] for t in glitch_list], dtype=np.float32)
    
    state_glitch = np.abs(glitch_list) > 0
    
    
    while np.sum(state_glitch) > 0:
      
      for i in range(n_glitch):
        if not state_glitch[i]:
          continue
        t_glitch = glitch_list[i]
        # allow a shift of T_SHIFT sec
        shifted_glitch = np.array(
            [raw_obs[max(int(t_glitch + t_shift),0):min(int(t_glitch + npt + t_shift),npt_raw_obs)] 
             for t_shift in np.arange(-sps*T_SHIFT,sps*T_SHIFT+1,2)], dtype=np.float32)
        
        shifted_glitch = np.apply_along_axis(lambda x: x-np.linspace(np.mean(x[:20]),np.mean(x[-20:]),x.shape[0]),
                                             1, shifted_glitch.astype(np.float32))
        shifted_glitch = np.apply_along_axis(lambda x: x/abs(x).max(),
                                             1, shifted_glitch.astype(np.float32))
  
        shifted_glitch_rec=autoencoder.reconstruct(shifted_glitch)
        relative_E_error=np.divide(np.sum(np.square(shifted_glitch_rec-shifted_glitch),axis=1), 
                                   np.sum(np.square(shifted_glitch),axis=1))
        
        t_shift_peak = peakutils.indexes(1/relative_E_error, thres=.5, min_dist=80)
        
        assert t_shift_peak.shape[0] <= 1
        # plt.plot(shifted_glitch[t_shift,:],'r')
        if t_shift_peak.shape[0] > 0:
          # print('>>>>>>>>>shift',np.linspace(-T_SHIFT,T_SHIFT,int(T_SHIFT*sps*2+1))[t_shift_peak],' sec for glitch No.',i,'at',t_glitch)
          glitch_list[i] = int(t_glitch+np.arange(-sps*T_SHIFT,sps*T_SHIFT+1,2)[t_shift_peak])
      
      before_deglitch = np.array(
          [raw_obs[glitch_list[i_glitch] + range(npt)] for i_glitch in
           range(glitch_list.shape[0])]
          , dtype=np.float32)
      
      # normalize to [-1.,1.]
      before_deglitch = np.apply_along_axis(lambda x: x-np.linspace(np.mean(x[:20]),np.mean(x[-20:]),x.shape[0]),
                                   1, before_deglitch)
      X = np.apply_along_axis(lambda x: x/abs(x).max(),
                                   1, before_deglitch.astype(np.float32))
      amp = np.apply_along_axis(lambda x: abs(x).max(), 1, before_deglitch)
      
      Y = autoencoder.reconstruct(X)
      cross_correlation = (Y * X).sum(axis=1) / np.sqrt(
          np.sum(Y ** 2, axis=1) * np.sum(X ** 2, axis=1))
      
      # subtract (reconstructed pure glitch) * (normalizing factor) from raw data

      for i_glitch in range(glitch_list.shape[0]):

          # -- remove the later glitch first
          # if not the last one .and. the next one is too close .and. the next one has not been removed
          
          if (not (i_glitch == (n_glitch - 1))) and (
                  ((glitch_list[i_glitch + 1] - glitch_list[i_glitch]) < 20 * 30) and
                  state_glitch[i_glitch + 1]):
            
              # print('multiglitch at ',glitch_list[min(i_glitch + 1, glitch_list.shape[0] - 1)])
              continue
            
          if state_glitch[i_glitch]:
              # omit poorly fitted ones
              if cross_correlation[i_glitch] < 0.6:
                state_glitch[i_glitch] = False
              else:
                raw_obs[glitch_list[i_glitch]+np.arange(npt)] -= Y[i_glitch, :] * amp[i_glitch]
                if PLOT_FIGURE:
                  reconstructed_glitch[i_glitch, :] = raw_obs[glitch_list[i_glitch]] + Y[i_glitch, :] * amp[i_glitch]
                  tshift_plot[i_glitch]=glitch_list[i_glitch]
                  
    st[i_component].data = raw_obs
  
  
  
    if PLOT_FIGURE:
      deglitched_obs=raw_obs.copy()
      # --------------------------------------------------------
      #%% ----  fig: raw data aligned with glitch free data ----
      # --------------------------------------------------------
      time_axis = np.linspace(0, (npt_raw_obs-1) / sps, npt_raw_obs)
      st = read_large_mseed('Input'+'/'+filename)
      raw_obs = st[i_component].data
      before_deglitch = np.array(
            [raw_obs[tshift_plot[i_glitch]+np.arange(npt)] for i_glitch in
             range(glitch_list.shape[0])]
            , dtype=np.float32)
      plt.figure(figsize=(24, 8))
      plt.subplots_adjust(left=0.05, bottom=0.1, right=0.99, top=0.99, hspace=0.1, wspace=0.1)
      
      plt.plot(time_axis, raw_obs, color='k', lw=1)
      for i_glitch in range(glitch_list.shape[0]):
          plt.plot(time_axis[tshift_plot[i_glitch]+np.arange(npt)],
                   before_deglitch[i_glitch, :],
                   color='red')
          plt.plot(time_axis[tshift_plot[i_glitch]+np.arange(npt)],
                   deglitched_obs[tshift_plot[i_glitch]+np.arange(npt)],
                   'g', lw=1)
          plt.plot(time_axis[tshift_plot[i_glitch]+np.arange(npt)],
                   reconstructed_glitch[i_glitch, :],
                   'b:')
          plt.scatter(time_axis[tshift_plot[i_glitch]], before_deglitch[i_glitch, 0], marker='o',
                      color='blue')
          plt.scatter(time_axis[tshift_plot[i_glitch] + npt], before_deglitch[i_glitch, -1], marker='x',
                      color='blue')
      plt.plot(time_axis, deglitched_obs - 6 * raw_obs.std(), 
               'g', lw=1)
      
      plt.legend(['raw data','before deglitch','glitch free data','reconstruction',])
      plt.ylabel('Raw unit', fontsize=16)
      plt.xlabel('Time (s)', fontsize=16)
      plt.xlim((0, time_axis[-1]))
      plt.show()
      
  
  st.write('Output/deglitched_%s' % filename)

print(time()-t1)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    