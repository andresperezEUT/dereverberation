"""
Perform a MAR estimation, then use the dereverberated signal to estimate
the IR through System IDentification, then compute the rt60 from the SID IR

results:
- rt60 estimation method works fine
- dereverberation works kind of fine
- due to high noise level on the estimated IR, better use edt estimator
"""


# %%
# import numpy as np


from ctf.ctf_methods import sid_stft2, rt60_bands, compute_t60
from blind_rt60.datasets import get_audio_files
from methods import *

import os
import matplotlib.pyplot as plt
import librosa
import scipy.signal
import copy
import librosa.display
import numpy as np

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



# %% PARAMETERS

#### parameters from "Online Dereverberation..."

sh_order = 0
dimM = (sh_order+1)**2

fs = 8000
window_size = 128 # samples
hop = 1/2 # in terms of windows
window_overlap = window_size*(1-hop)
nfft = window_size
dimK = nfft//2+1

if hop == 1:
    window_type = 'boxcar'
else:
    window_type = 'hann'

audio_file_length = 20.  ## seconds
audio_file_length_samples = int(audio_file_length * fs)
audio_file_offset = 5. ## seconds
audio_file_offset_samples = int(audio_file_offset * fs)

## MAR-------
# D = int(np.floor(ir_start_time * fs / window_overlap)) # ir start frame
tau = int(1/hop)
print('tau=',tau)
if tau < 1:
    raise Warning('D should be at least 1!!!')

# noise_power = 1e-5

# get audio files
num_files = 7
dataset = 'DSD100'
subset = ''
main_path = '/Volumes/Dinge/datasets'
audio_files = get_audio_files(main_path, dataset, subset)[:num_files]
# valid_audio_file_idx = np.array([0, 1, 2, 4, 5, 6, 8, 9, 11, 12, 15, 16, 18, 19, 20, 22, 23,
#                                  24, 29, 31, 32, 33, 34, 35, 37, 38, 39, 41, 42, 43, 44, 46, 47, 48,
#                                  49, 51, 52, 53, 54, 56, 58, 59, 60, 63, 65, 67, 68, 69, 72, 74,
#                                  77, 79, 80, 81, 83, 84, 85, 86, 87, 88, 89, 91, 92, 94, 95, 96, 98])
# audio_files = np.array(audio_files)[valid_audio_file_idx]
# num_files = len(audio_files)
N = num_files


# %%
# FREQUENCY BANDS
freqs = np.arange(dimK)*fs/2/(dimK-1) # frequency indices to Hz

nBands = 1
band_centerfreqs = np.empty(nBands)
band_centerfreqs[0] = 1000
for nb in range(1, nBands):
    band_centerfreqs[nb] = 2 * band_centerfreqs[nb - 1]



# %% IRS

# TODO: real uniform along the sphere
azi = np.random.rand()*2*np.pi
incl = np.random.rand()*np.pi
# azi = 0
# incl = np.pi/2

print('AZI - ELE', azi, np.pi/2 - incl)
dirs = np.asarray([[azi, incl]])
basisType = 'real'
y = masp.get_sh(1, dirs, basisType) * np.sqrt(4*np.pi) * [1, 1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)] ## ACN, SN3D

rt60_decay = 0.05  # % of rt60_0 per octave
rt60_vector = np.asarray([0.4])

I = 11  # number of repetitions

rt_methods = ['edt', 't10', 't20', 't30']
R = len(rt_methods)

rt60_1kHz = np.zeros(I) # true rt60 at 1kHz
rt60_ir_true = np.zeros((I,R))
rt60_ir_est_true = np.zeros((I,N,R))
rt60_ir_est_derv = np.zeros((I,N,R))

L = fs # todo: adjust here filter length
ir_true = np.zeros((I, L))
ir_est_true = np.zeros((I, N, L))
ir_est_derv = np.zeros((I, N, L))



for iter in range(I):

    print('--------------------------------------------')
    print('ITER:', iter)
    rt60_0 = np.random.rand() * 0.6 + 0.4

    room = np.array([10.2, 7.1, 3.2])
    rt60 = rt60_bands(rt60_0, nBands, rt60_decay)

    abs_wall = srs.find_abs_coeffs_from_rt(room, rt60)[0]

    # Critical distance for the room
    _, d_critical, _ = srs.room_stats(room, abs_wall, verbose=False)

    # Receiver position
    rec = (room/2)[np.newaxis] # center of the room
    nRec = rec.shape[0]

    # d_critical distance with defined angular position
    azi = np.random.rand() * 2 * np.pi
    incl = np.random.rand() * np.pi

    azi = azi + np.pi # TODO: fix in srs library!!!
    src_sph = np.array([azi, np.pi/2-incl, d_critical.mean()/2])
    src_cart = masp.sph2cart(src_sph)
    src = rec + src_cart
    nSrc = src.shape[0]

    # SH orders for receivers
    rec_orders = np.array([sh_order])

    maxlim = 1.  # just stop if the echogram goes beyond that time ( or just set it to max(rt60) )
    limits = np.ones(nBands)*maxlim # hardcoded!

    abs_echograms = srs.compute_echograms_sh(room, src, rec, abs_wall, limits, rec_orders)
    irs = srs.render_rirs_sh(abs_echograms, band_centerfreqs, fs).squeeze().T
    if irs.ndim == 1:
        irs = irs[np.newaxis, :]
    # Normalize as SN3D
    irs *= np.sqrt(4 * np.pi)



    # %% SYNTHESIZE AUDIOS

    for n, af in enumerate(audio_files):
        start_sample = audio_file_offset_samples
        end_sample = audio_file_offset_samples + audio_file_length_samples
    # # Open audio files and encode into ambisonics

        mono_s_t = librosa.core.load(af, sr=fs, mono=True)[0][start_sample:end_sample]
        # mono_s_t = np.random.normal(size=(audio_file_length_samples))

        s_t = mono_s_t
        f, t, s_tf = scipy.signal.stft(s_t, fs, window=window_type, nperseg=window_size, noverlap=window_overlap, nfft=nfft)

        # Get reverberant signal
        y_t = np.zeros((dimM, audio_file_length_samples))  # reverberant signal
        for m in range(dimM):
            y_t[m] = scipy.signal.fftconvolve(mono_s_t, irs[m])[:audio_file_length_samples]  # keep original length

        # Add some noise... TODO

        f, t, y_tf = scipy.signal.stft(y_t, fs, window=window_type, nperseg=window_size, noverlap=window_overlap, nfft=nfft)
        dimM, dimK, dimN = y_tf.shape


        # %% ANALYSIS

        # parameters
        p = 0.25
        i_max = 10
        ita = 1e-4
        epsilon = 1e-8
        L = 20  # number of frames for the IIR filter

        norm_s_tf = np.linalg.norm(s_tf[0])
        norm_y_tf = np.linalg.norm(y_tf[0])

        # estimate
        est_s_tf, C, _ = estimate_MAR_sparse_parallel(y_tf, L, tau, p, i_max, ita, epsilon)
        _, est_s_t = scipy.signal.istft(est_s_tf, fs, window=window_type, nperseg=window_size, noverlap=window_overlap, nfft=nfft)
        est_s_t = est_s_t[:, :audio_file_length_samples]

        # plot_magnitude_spectrogram(s_tf, fs)
        # plot_magnitude_spectrogram(y_tf, fs)
        # plot_magnitude_spectrogram(est_s_tf, fs)
        # plt.show()

        # coefs over frequency
        # plt.figure()
        # plt.plot(np.abs(C.squeeze()))
        # plt.show()


        # %% AKIS STFT SYSTEM IDENTIFICATION


        ir_true[iter] = irs[0]  # true impulse response
        filtersize = ir_true[iter].size # true, groundtruth
        winsize = 8 * filtersize
        hopsize = winsize / 16

        # IR ESTIMATION FROM TRUE ANECHOIC SIGNAL
        ir_est_true[iter, n] = sid_stft2(s_t, y_t[0], winsize, hopsize, filtersize)

        # IR ESTIMATION FROM ESTIMATED ANECHOIC SIGNAL
        ir_est_derv[iter, n] = sid_stft2(est_s_t[0], y_t[0], winsize, hopsize, filtersize)

        # plt.figure()
        # plt.plot(ir_true[iter])
        # plt.plot(ir_est_true[iter], linestyle='--', label='true')
        # plt.plot(ir_est_derv[iter], linestyle='--', label='derv')
        # # plt.plot(np.abs(ir_est-ir), linestyle=':')
        # plt.title('SID with non-overlapped STFT - true signal')
        # plt.show()


        # %% t60 estimation

        plot = False
        true = compute_t60(ir_true[iter], fs, band_centerfreqs, plot=plot, title='True IR')
        est_true = compute_t60(ir_est_true[iter, n], fs, band_centerfreqs, plot=plot, title='Estimated IR, true signal')
        est_derv = compute_t60(ir_est_derv[iter, n], fs, band_centerfreqs, plot=plot, title='Estimated IR, derv signal')

        # Report values
        print('--true rt60, true IR--')
        print(rt60[0])
        print('--estimated rt60, true IR--')
        print(true)
        print('--estimated rt60, true anechoic signal --')
        print(est_true)
        print('--estimated rt60, dereverberated signal --')
        print(est_derv)

        # Save values (only one band)
        rt60_1kHz[iter] = rt60[0]
        rt60_ir_true[iter] = true.squeeze()
        rt60_ir_est_true[iter,n] = est_true.squeeze()
        rt60_ir_est_derv[iter,n] = est_derv.squeeze()




# save

# np.save('/Users/andres.perez/source/dereverberation/ctf/ctf_mar_exp2/ir_true'+'_'+str(sh_order),ir_true)
# np.save('/Users/andres.perez/source/dereverberation/ctf/ctf_mar_exp2/ir_est_true'+'_'+str(sh_order),ir_est_true)
# np.save('/Users/andres.perez/source/dereverberation/ctf/ctf_mar_exp2/ir_est_derv'+'_'+str(sh_order),ir_est_derv)
# np.save('/Users/andres.perez/source/dereverberation/ctf/ctf_mar_exp2/rt60_1kHz'+'_'+str(sh_order),rt60_1kHz)
# np.save('/Users/andres.perez/source/dereverberation/ctf/ctf_mar_exp2/rt60_ir_true'+'_'+str(sh_order),rt60_ir_true)
# np.save('/Users/andres.perez/source/dereverberation/ctf/ctf_mar_exp2/rt60_ir_est_true'+'_'+str(sh_order),rt60_ir_est_true)
# np.save('/Users/andres.perez/source/dereverberation/ctf/ctf_mar_exp2/rt60_ir_est_derv'+'_'+str(sh_order),rt60_ir_est_derv)


# %% results


iii = np.argsort(rt60_1kHz)
plt.figure()
for n in range(N):

    plt.subplot(N,3,(3*n)+1)
    plt.title('Estimated RT60, True IR')
    plt.plot(np.arange(I), rt60_1kHz[iii], '-o', label='true')
    for rt_method_idx, rt_method in enumerate(rt_methods):
        plt.plot(np.arange(I), rt60_ir_true[iii, rt_method_idx], '-o', label=rt_method)

    plt.subplot(N,3,(3*n)+2)
    plt.title('Estimated rt60, true anechoic signal')
    plt.plot(np.arange(I), rt60_1kHz[iii], '-o', label='true')
    for rt_method_idx, rt_method in enumerate(rt_methods):
        plt.plot(np.arange(I), rt60_ir_est_true[iii, n, rt_method_idx], '-o', label=rt_method)
    plt.legend()
    plt.xlabel('Iteration number')
    plt.ylabel('RT60')

    plt.subplot(N, 3, (3 * n) + 3)
    plt.title('Estimated rt60, dereverberated signal')
    plt.plot(np.arange(I), rt60_1kHz[iii], '-o', label='true')
    for rt_method_idx, rt_method in enumerate(rt_methods):
        plt.plot(np.arange(I), rt60_ir_est_derv[iii, n, rt_method_idx], '-o', label=rt_method)

# %%
rt_method_idx = 1

iii = np.argsort(rt60_ir_true[:, rt_method_idx])

plt.figure()
plt.title('RT60 estimationm average of 5 sources, FOA')
# plt.plot(np.arange(I), rt60_1kHz[iii], '-o', markersize=2, label='true RT')
plt.plot(np.arange(I), rt60_ir_true[iii, rt_method_idx], '--o', markersize=2, label='measured RT, true IR')
plt.errorbar(np.arange(I),
             np.mean(rt60_ir_est_true[iii, :, rt_method_idx], axis=1),
             yerr=np.std(rt60_ir_est_true[iii, :, rt_method_idx], axis=1),
             linestyle=':', markersize=2, label='measured RT, oracle SID ')
plt.errorbar(np.arange(I),
             np.mean(rt60_ir_est_derv[iii, :, rt_method_idx], axis=1),
             yerr=np.std(rt60_ir_est_derv[iii, :, rt_method_idx], axis=1),
             linestyle=':', markersize=2, label='measured RT, dereverberated SID')
plt.legend()




# %%

rt_method_idx = 1
mean_rt60_ir_est_derv = np.mean(rt60_ir_est_derv[:, :, rt_method_idx], axis=1)
std_rt60_ir_est_derv = np.std(rt60_ir_est_derv[:, :, rt_method_idx], axis=1)

plt.figure()
plt.plot(np.arange(I), rt60_1kHz[iii], '-o', label='true')
plt.errorbar(np.arange(I), mean_rt60_ir_est_derv[iii], yerr=std_rt60_ir_est_derv[iii], fmt='-o', markersize=2)



def line(x, m, n):
    return m * x + n
popt, pcov = scipy.optimize.curve_fit(line, mean_rt60_ir_est_derv[iii], rt60_1kHz[iii], sigma=std_rt60_ir_est_derv[iii], absolute_sigma=True)

print("popt, pcov, joint std")
print('----------------------------------------')
print(popt)
print(pcov)
var = np.sum(np.diag(pcov))
std = np.sqrt(var) # joint standard deviation is sqrt of sum of variances https://socratic.org/statistics/random-variables/addition-rules-for-variances
print(std)

plt.plot(np.arange(I), mean_rt60_ir_est_derv[iii]*popt[0]+popt[1], linestyle='--')

