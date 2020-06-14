"""
esperiment 1 - DATA ANALYSIS

DSD set, seconds 1-5, 8 kHz, L=5, imax=10, w=128 0.4-1 s rt
save data in /ctf_data
"""

import numpy as np
from ctf_methods import stft, istft, compute_t60, rt60_bands
from methods import *
import os
import matplotlib.pyplot as plt
import librosa
import scipy.signal
import copy
import librosa.display

import sys
pp = "/Users/andres.perez/source/masp"
sys.path.append(pp)
import masp
from masp import shoebox_room_sim as srs
pp = "/Users/andres.perez/source/parametric_spatial_audio_processing"
sys.path.append(pp)
import parametric_spatial_audio_processing as psa

cmap = plt.get_cmap("tab20c")
plt.style.use('seaborn-whitegrid')

from blind_rt60.datasets import get_audio_files
from ctf.ctf_methods import sid_stft2, compute_t60


# %% PARAMETERS

sr = 8000
sh_order = 0
dimM = (sh_order+1)**2

sr = 8000
window_size = 128 # samples
hop = 1/2 # in terms of windows
window_overlap = int(window_size*(1-hop))
nfft = window_size
dimK = nfft//2+1

num_files = -1
dataset = 'DSD100'
subset = ''
main_path = '/Volumes/Dinge/datasets'
audio_files = get_audio_files(main_path, dataset, subset)[:num_files]
valid_audio_file_idx = np.array([0, 1, 2, 4, 5, 6, 8, 9, 11, 12, 15, 16, 18, 19, 20, 22, 23,
                                 24, 29, 31, 32, 33, 34, 35, 37, 38, 39, 41, 42, 43, 44, 46, 47, 48,
                                 49, 51, 52, 53, 54, 56, 57, 58, 59, 60, 63, 65, 67, 68, 69, 72, 74,
                                 77, 79, 80, 81, 83, 84, 85, 86, 87, 88, 89, 91, 92, 94, 95, 96, 98])
audio_files = np.array(audio_files)[valid_audio_file_idx]

nBands = 1
band_centerfreqs = np.empty(nBands)
band_centerfreqs[0] = 1000
for nb in range(1, nBands):
    band_centerfreqs[nb] = 2 * band_centerfreqs[nb - 1]


rt60_0s = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ir_length_seconds =  0.125
ir_length_samples = int(ir_length_seconds*sr)
rt_methods = ['edt', 't10', 't20', 't30']
Ls = [5]  # number of frames for the IIR filter
R = len(rt60_0s)
N = len(audio_files)

tau = max(int(1 / hop), 1)


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def index_of_last_element(array, cond):
    return np.nonzero(array > cond)[0][-1]


# %% ANALYSIS

# SHAPE: ((R, N, len(Ls), dimM, ir_length_samples))
irs_estimated = np.load('/Users/andres.perez/source/dereverberation/ctf/ctf_data/irs_estimated_'+str(sh_order)+'_1000.npy')



# %% ANALYSIS
af_idx = 50
m_idx = 0
l_idx = 0
nb = 0
band = band_centerfreqs[nb]
flim = [band / np.sqrt(2), band * np.sqrt(2)]
sos = scipy.signal.butter(3, flim, btype='bandpass', output='sos', fs=sr)

L = irs_estimated.shape[-1]
plt.figure()

for rt_idx, rt in enumerate(rt60_0s):
    true_rt60 = rt60_0s[rt_idx]
    ir = irs_estimated[rt_idx, af_idx, l_idx, m_idx]
    ir = ir[tau * window_overlap:]
    filtered_ir = scipy.signal.sosfilt(sos, ir)
    a_t = np.abs(scipy.signal.hilbert(filtered_ir))
    sch = np.cumsum(a_t[::-1] ** 2)[::-1]
    sch_db = 10.0 * np.log10(sch / np.max(sch))
    sch_db = sch_db[:int(0.9*sch_db.size)]
    m = -10 / index_of_last_element(sch_db, -10)  # dB / sample
    rt60_value = -60 / m / sr

    print('-----------------------------')
    print('true:      ', true_rt60)
    print('estimated: ', rt60_value)

    x = np.arange(len(filtered_ir))
    m_true = -60 / true_rt60 / sr
    plt.plot(x,x*m_true, label='true '+str(rt), linestyle='-.',)

    plt.plot(sch_db, label='sch_db '+str(rt), color=colors[rt_idx])
    plt.plot(x, m * x , label='edt'+str(rt), linestyle='--', linewidth=1, color=colors[rt_idx])
plt.legend()


rt_idx = 0
rt = rt60_0s[rt_idx]
plt.figure()


true_rt60 = rt60_0s[rt_idx]
ir = irs_estimated[rt_idx, af_idx, l_idx, m_idx]
ir = ir[tau * window_overlap:]
filtered_ir = scipy.signal.sosfilt(sos, ir)
a_t = np.abs(scipy.signal.hilbert(filtered_ir))
sch = np.cumsum(a_t[::-1] ** 2)[::-1]
sch_db = 10.0 * np.log10(sch / np.max(sch))
sch_db = sch_db[:int(0.9 * sch_db.size)]

plt.plot(a_t, label=str(rt), linewidth=1, color=colors[0])
plt.plot(sch, label=str(rt), linestyle='--', linewidth=1, color=colors[1])

# plt.figure()
# plt.plot(ir)


m_idx = 0
l_idx = 0
nb = 0
band = band_centerfreqs[nb]
flim = [band / np.sqrt(2), band * np.sqrt(2)]
sos = scipy.signal.butter(3, flim, btype='bandpass', output='sos', fs=sr)

def fit_rt60(data, sr):
    rt_methods = ['edt', 't10', 't20', 't30']
    rt60 = np.empty(len(rt_methods))
    for rt_method_idx, rt_method in enumerate(rt_methods):
        if rt_method == 'edt':
            m = -10 / index_of_last_element(data, -10)  # dB / sample
            n = 0  # offset
        elif rt_method == 't10':
            a = index_of_last_element(data, -5)
            b = index_of_last_element(data, -15)
            m = (15 - 5) / (a - b)
            n = -5 - m * a  # - n = mx + y
        elif rt_method == 't20':
            a = index_of_last_element(data, -5)
            b = index_of_last_element(data, -25)
            m = (25 - 5) / (a - b)
            n = -5 - m * a  # - n = mx + y
        elif rt_method == 't30':
            a = index_of_last_element(data, -5)
            b = index_of_last_element(data, -35)
            m = (35 - 5) / (a - b)
            n = -5 - m * a  # - n = mx + y
        rt60_value = (-60 - n) / m / sr
        rt60[rt_method_idx] = rt60_value
    return rt60


# shape: (r, method)
estimated_rt60 = np.empty((R, 4))

L = irs_estimated.shape[-1]


for rt_idx, rt in enumerate(rt60_0s):
    true_rt60 = rt60_0s[rt_idx]
    ir = irs_estimated[rt_idx, af_idx, l_idx, m_idx]
    ir = ir[tau * window_overlap:]
    filtered_ir = scipy.signal.sosfilt(sos, ir)
    a_t = np.abs(scipy.signal.hilbert(filtered_ir))
    sch = np.cumsum(a_t[::-1] ** 2)[::-1]
    sch_db = 10.0 * np.log10(sch / np.max(sch))
    sch_db = sch_db[:int(0.9*sch_db.size)]

    fit_values = fit_rt60(sch_db,sr)
    estimated_rt60[rt_idx] = fit_values

rt_methods = ['edt', 't10', 't20', 't30']
plt.figure()
plt.plot(rt60_0s, rt60_0s, linestyle='--', linewidth=1, color=colors[0])
for method_idx, method in enumerate(rt_methods):
    plt.plot(rt60_0s, estimated_rt60[:,method_idx], label=method, linewidth=1, color=colors[method_idx+1])
plt.legend()
