# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 22:01:38 2021

@author: xwccc
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 13:43:28 2021

@author: xwccc
"""
from obspy import UTCDateTime,read
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import matplotlib

# def read_mps_catalog():
    
# cata = r'C:\Users\xwccc\Desktop\ucla\UCLA_v4\glitches_night1.txt'
# with open(cata,'r') as f:
#     lines = f.readlines()
# lines = lines[3:-39]
# starttime = UTCDateTime('2019-07-01T00:00:00')
# times = []
# for i in range(len(lines)):
#     tmp = lines[i].split()
#     start = UTCDateTime(tmp[1])
#     comp = tmp[6]
#     if comp == '1':
#         print(start-starttime)
#         times.append(start-starttime)

# cata = r'C:\Users\xwccc\Desktop\ucla\UCLA_v4\glitches_day.txt'
# with open(cata,'r') as f:
#     lines = f.readlines()
# lines = lines[3:-39]
# for i in range(len(lines)):
#     tmp = lines[i].split()
#     start = UTCDateTime(tmp[1])
#     comp = tmp[6]
#     if comp == '1':
#         print(start-starttime)
#         times.append(start-starttime)
        
# cata = r'C:\Users\xwccc\Desktop\ucla\UCLA_v4\glitches_night2.txt'
# with open(cata,'r') as f:
#     lines = f.readlines()
# lines = lines[3:-39]
# for i in range(len(lines)):
#     tmp = lines[i].split()
#     start = UTCDateTime(tmp[1])
#     comp = tmp[6]
#     if comp == '1':
#         print(start-starttime)
#         times.append(start-starttime)


## MPS
def read_mps_catalog(cata,comp,starttime):
    times = []
    clist = 'UVW'
    cnum = '567'
    cnum = int(cnum[clist.index(comp)])
    with open(cata,'r') as f:
        lines = f.readlines()
    lines = lines[3:-39]
    for i in range(len(lines)):
        tmp = lines[i].split()
        start = UTCDateTime(tmp[1])
        comp = tmp[cnum]
        if comp == '1':
            # print(start-starttime)
            times.append(start-starttime)
    times = [i+3 for i in times]
    return np.array(times)
    

#### UCLA
def read_ucla_catalog(cata,starttime):
    utime = []
    with open(cata) as f:
        lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i]
        t0 = UTCDateTime(line.split()[0])
        utime.append(t0-starttime)
    return np.array(utime)

### Manual
def read_manual_pick(cata):
    htime = np.load(cata)[:,0]/20
    return htime

def butter_bandpass_filtfilt(data,cutoff,fs,order=4):
    if data.ndim == 2:
        axis = 1
    else: 
        axis = -1
    wn = 2*cutoff/fs
    b, a = signal.butter(order, wn, 'bandpass', analog = False)
    output = signal.filtfilt(b, a, data, axis=axis)
    return output 
def butter_lowpass_filtfilt(data,cutoff,fs,order=4):
    wn = 2*cutoff/fs
    b, a = signal.butter(order, wn, 'lowpass', analog = False)
    output = signal.filtfilt(b, a, data)
    return output 
# for comparison
def butter_highpass_filtfilt(data,cutoff,fs,order=4):
    if data.ndim == 2:
        axis = 1
    else: 
        axis = -1
    wn = 2*cutoff/fs
    b, a = signal.butter(order, wn, 'highpass', analog = False)
    output = signal.filtfilt(b, a, data, axis=axis)
    return output
def does_pick_exist(array,pick):
    exist = 0
    for i in range(len(array)):
        if abs(array[i]-pick) <= 5:
            exist = 1
    return exist

def find_corresponding_pick(htime,timelist,tr):
    cor = np.zeros(len(htime))
    for i in range(len(htime)):
        cor[i] = does_pick_exist(timelist,htime[i])
    plt.plot(tr.times(),tr.data)
    plt.scatter(htime[cor>0],np.ones(len(htime[cor>0]))*1000,color ='r')
    plt.scatter(htime[cor<1],np.ones(len(htime[cor<1]))*2000,color ='b')


day = '0701'
comp = 'V'
starttime =  UTCDateTime('2019-07-01T00:00:00')
mcata = r'C:\Users\xwccc\Desktop\ucla\manual\glitches_%s.txt' %day
ucata = r'C:\Users\xwccc\Desktop\ucla\manual\%s%s.txt' %(day,comp)
hcata = r'C:\Users\xwccc\Desktop\ucla\manual\%sBH%s.npy' %(day,comp)
file = r'C:\Users\xwccc\Desktop\ucla\manual\%s.mseed' %day

mtime = read_mps_catalog(mcata,comp,starttime)  
utime = read_ucla_catalog(ucata,starttime) 
htime = read_manual_pick(hcata)
gtime = idx/20
st = read(file)
clist = 'UVW'
tr = st[[0,1,2][clist.index(comp)]]
mseed = 'C:\\Users\\xwccc\\Desktop\\ucla\\UCLA_v4\\3comp0701.mseed'
st = read(mseed)
tr = st[1]
data = tr.data
# data = butter_bandpass_filtfilt(data,np.array([0.001,2]),20,2)
# data = butter_lowpass_filtfilt(data,2,20,4)
# data = butter_highpass_filtfilt(data,0.005,20,4)
data = butter_bandpass_filtfilt(data,np.array([0.001,2]),20,2)
filenm = 'C:\\Users\\xwccc\\Desktop\\ucla\\UCLA_v4\\0701BHV.npy'
htime = np.load(filenm)[:,0]/20
encoderfile = 'C:\\Users\\xwccc\\Desktop\\ucla\\UCLA_v4\\glitch_list_0701_autoencoder2.npy'
etime = np.load(encoderfile)
etime = etime/20



fig=plt.figure(tight_layout=True,figsize=[13,7])
plt.subplot(211)

plt.plot(tr.times(),data,linewidth=0.4,color = 'k')
s0=6
s1=8
plt.scatter(htime,np.ones(len(htime))+11500,s=s0,color='r',label='manual pick')   # human
plt.scatter(mtime,np.ones(len(mtime))+10000,s=s0,color='b',label='MPS')   # mps
plt.scatter(gtime,np.ones(len(gtime))+8500,s=s0,color='g',label='CNN')  # cnn
plt.scatter(utime,np.ones(len(utime))+7000,s=s0,color='y',label='UCLA')  # ucla
plt.scatter(etime+8,np.ones(len(etime))+5500,s=s0,color='c',label='AE') # encoder
plt.text(30170,12800,'(a)',fontdict={'family' : 'Times New Roman', 'size'   : 20})
# plt.xlabel('time (s)',fontdict={'family' : 'Times New Roman', 'size'   : 14})
plt.ylabel('velocity (counts)',fontdict={'family' : 'Times New Roman', 'size'   : 18})
plt.xlim([30000, 58000])
plt.ylim([-13000,15000])
plt.vlines(30800, -6000, 12000,colors = 'c')
plt.vlines(31600, -6000, 12000,colors = 'c')
plt.hlines(-6000,30800,31600,colors = 'c')
plt.hlines(12000,30800,31600,colors = 'c')

plt.vlines(49500, -12000, 12000,colors = 'c')
plt.vlines(51000, -12000, 12000,colors = 'c')
plt.hlines(-12000,49500,51000,colors = 'c')
plt.hlines(12000,49500,51000,colors = 'c')

# plt.legend([l1,l2,l3,l4,l5,l6],['data','manual pick','MPS','CNN','UCLA','AE'])
font1={'family' : 'Times New Roman', 'size'   : 14,'weight':'bold'}
font2={'family' : 'Times New Roman', 'size'   : 18}
# plt.text(87000,11300,'manual pick',fontdict=font1,color='r')
# plt.text(87000,9600,'MPS',fontdict=font1,color='b')
# plt.text(87000,8100,'CNN',fontdict=font1,color='g')
# plt.text(87000,6500,'UCLA',fontdict=font1,color='y')
# plt.text(87000,5200,'AE',fontdict=font1,color='c')
# plt.legend(prop=font1,loc = 'lower right',bbox_to_anchor=(1.015, -0.03),framealpha=0.4)
plt.title('20190701 BHV',fontdict={'family' : 'Times New Roman', 'size'   : 20})
ax = plt.gca()
plt.tick_params(labelsize=18)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax.yaxis.get_major_formatter().set_powerlimits((0,1))
# ax.yaxis.get_offset_text().set_fontsize(14)
# ax.yaxis.get_offset_text().set_fontfamily(font2)
ax.yaxis.get_offset_text().set_fontname('Times New Roman')
ax.yaxis.get_offset_text().set_fontsize(18)
plt.subplot(223)
plt.plot(tr.times(),data,linewidth=0.5,color = 'k')
plt.scatter(htime,np.ones(len(htime))+9500,s=s1,color='r',label='manual pick')   # human
plt.scatter(mtime,np.ones(len(mtime))+8500,s=s1,color='b',label='MPS')   # mps
plt.scatter(gtime,np.ones(len(gtime))+7500,s=s1,color='g',label='CNN')  # cnn
plt.scatter(utime,np.ones(len(utime))+6500,s=s1,color='y',label='UCLA')  # ucla
plt.scatter(etime+8,np.ones(len(etime))+5500,s=s1,color='c',label='AE') # encoder
plt.xlabel('time (s)',fontdict={'family' : 'Times New Roman', 'size'   : 18})
# plt.ylabel('velocity (counts)',fontdict={'family' : 'Times New Roman', 'size'   : 18})
# plt.legend([l1,l2,l3,l4,l5,l6],['data','manual pick','MPS','CNN','UCLA','AE'])
font1={'family' : 'Times New Roman', 'size'   : 18}
# plt.legend(prop=font1,loc = 'lower right')
# plt.title('20190701 BHV',fontdict={'family' : 'Times New Roman', 'size'   : 24})
# plt.plot(tr.times(),tr.data,linewidth=0.5)
# plt.scatter(peak_time,peak_amp,color = 'r')
# # plt.scatter(peak_time,gli_amp[gli_amp>200]*np.array([np.sign(data[int(x*20)]) for x in peak_time]),color = 'r')
plt.xlim([30800,31600])
plt.ylim([-6000,12000])
plt.text(30810,10550,'(b)',fontdict={'family' : 'Times New Roman', 'size'   : 20})
ax = plt.gca()
plt.tick_params(labelsize=18)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax.yaxis.get_major_formatter().set_powerlimits((0,1))
ax.yaxis.get_offset_text().set_fontname('Times New Roman')
ax.yaxis.get_offset_text().set_fontsize(18)
plt.subplot(224)
plt.plot(tr.times(),data,linewidth=0.5,color = 'k')
plt.scatter(htime,np.ones(len(htime))+9500,s=s1,color='r',label='manual pick')   # human
plt.scatter(mtime,np.ones(len(mtime))+8500,s=s1,color='b',label='MPS')   # mps
plt.scatter(gtime,np.ones(len(gtime))+7500,s=s1,color='g',label='CNN')  # cnn
plt.scatter(utime,np.ones(len(utime))+6500,s=s1,color='y',label='UCLA')  # ucla
plt.scatter(etime+8,np.ones(len(etime))+5500,s=s1,color='c',label='AE') # encoder
plt.xlabel('time (s)',fontdict={'family' : 'Times New Roman', 'size'   : 18})
# plt.ylabel('velocity (counts)',fontdict={'family' : 'Times New Roman', 'size'   : 18})
# plt.legend([l1,l2,l3,l4,l5,l6],['data','manual pick','MPS','CNN','UCLA','AE'])
font1={'family' : 'Times New Roman', 'size'   : 18}
# plt.legend(prop=font1,loc = 'lower right')
# plt.title('20190701 BHV',fontdict={'family' : 'Times New Roman', 'size'   : 24})
# plt.plot(tr.times(),tr.data,linewidth=0.5)
# plt.scatter(peak_time,peak_amp,color = 'r')
# # plt.scatter(peak_time,gli_amp[gli_amp>200]*np.array([np.sign(data[int(x*20)]) for x in peak_time]),color = 'r')
plt.xlim([49500,51000])
plt.ylim([-12000,12000])
plt.text(49520,10100,'(c)',fontdict={'family' : 'Times New Roman', 'size'   : 20})
ax = plt.gca()
plt.tick_params(labelsize=18)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax.yaxis.get_major_formatter().set_powerlimits((0,1))
ax.yaxis.get_offset_text().set_fontname('Times New Roman')
ax.yaxis.get_offset_text().set_fontsize(18)
# original
# plt.figure(tight_layout=True,figsize=[15,7])
# plt.subplot(211)
# plt.plot(tr.times(),data,linewidth=0.4,color = 'k')
# s0=6
# s1=8
# plt.scatter(htime,np.ones(len(htime))+7500,s=s0,color='r',label='manually pick')   # human
# plt.scatter(mtime,np.ones(len(mtime))+7000,s=s0,color='b',label='MPS')   # mps
# plt.scatter(gtime,np.ones(len(gtime))+6500,s=s0,color='g',label='CNN')  # cnn
# plt.scatter(utime,np.ones(len(utime))+6000,s=s0,color='y',label='UCLA')  # ucla
# plt.scatter(etime+8,np.ones(len(etime))+5500,s=s0,color='c',label='AE') # encoder
# plt.text(1000,8500,'(a)',fontdict={'family' : 'Times New Roman', 'size'   : 20})
# plt.xlabel('time (s)',fontdict={'family' : 'Times New Roman', 'size'   : 14})
# plt.ylabel('velocity (counts)',fontdict={'family' : 'Times New Roman', 'size'   : 14})
# plt.xlim([0, 86800])
# plt.ylim([-10000,10100])
# plt.vlines(63300, -1000, 10000,colors = 'c')
# plt.vlines(69700, -1000, 10000,colors = 'c')
# plt.hlines(-1000,63300,69700,colors = 'c')
# plt.hlines(10000,63300,69700,colors = 'c')
# # plt.legend([l1,l2,l3,l4,l5,l6],['data','manual pick','MPS','CNN','UCLA','AE'])
# font1={'family' : 'Times New Roman', 'size'   : 12}
# plt.legend(prop=font1,loc = 'lower right',bbox_to_anchor=(1.015, -0.03),framealpha=0.4)
# plt.title('20190701 BHV',fontdict={'family' : 'Times New Roman', 'size'   : 20})
# ax = plt.gca()
# plt.tick_params(labelsize=14)
# labels = ax.get_xticklabels() + ax.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# plt.subplot(212)
# plt.plot(tr.times(),data,linewidth=0.5,color = 'k')
# plt.scatter(htime,np.ones(len(htime))+7500,s=s1,color='r',label='manual pick')   # human
# plt.scatter(mtime,np.ones(len(mtime))+7000,s=s1,color='b',label='MPS')   # mps
# plt.scatter(gtime,np.ones(len(gtime))+6500,s=s1,color='g',label='CNN')  # cnn
# plt.scatter(utime,np.ones(len(utime))+6000,s=s1,color='y',label='UCLA')  # ucla
# plt.scatter(etime+8,np.ones(len(etime))+5500,s=s1,color='c',label='AE') # encoder
# plt.xlabel('time (s)',fontdict={'family' : 'Times New Roman', 'size'   : 14})
# plt.ylabel('velocity (counts)',fontdict={'family' : 'Times New Roman', 'size'   : 14})
# # plt.legend([l1,l2,l3,l4,l5,l6],['data','manual pick','MPS','CNN','UCLA','AE'])
# font1={'family' : 'Times New Roman', 'size'   : 14}
# # plt.legend(prop=font1,loc = 'lower right')
# # plt.title('20190701 BHV',fontdict={'family' : 'Times New Roman', 'size'   : 24})
# # plt.plot(tr.times(),tr.data,linewidth=0.5)
# # plt.scatter(peak_time,peak_amp,color = 'r')
# # # plt.scatter(peak_time,gli_amp[gli_amp>200]*np.array([np.sign(data[int(x*20)]) for x in peak_time]),color = 'r')
# plt.xlim([63300,69700])
# plt.ylim([-1000,10000])
# plt.text(63380,9000,'(b)',fontdict={'family' : 'Times New Roman', 'size'   : 20})
# ax = plt.gca()
# plt.tick_params(labelsize=14)
# labels = ax.get_xticklabels() + ax.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]