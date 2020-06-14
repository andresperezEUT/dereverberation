"""
esperiment 1

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


# get audio files
num_files = -1
dataset = 'DSD100'
subset = ''
main_path = '/Volumes/Dinge/datasets'
audio_files = get_audio_files(main_path, dataset, subset)[:num_files]

# for dsd100
valid_audio_file_idx = np.array([0, 1, 2, 4, 5, 6, 8, 9, 11, 12, 15, 16, 18, 19, 20, 22, 23,
                                 24, 29, 31, 32, 33, 34, 35, 37, 38, 39, 41, 42, 43, 44, 46, 47, 48,
                                 49, 51, 52, 53, 54, 56, 57, 58, 59, 60, 63, 65, 67, 68, 69, 72, 74,
                                 77, 79, 80, 81, 83, 84, 85, 86, 87, 88, 89, 91, 92, 94, 95, 96, 98])
audio_files = np.array(audio_files)[valid_audio_file_idx]

sh_order = 1
dimM = (sh_order+1)**2

sr = 8000
window_size = 128 # samples
hop = 1/2 # in terms of windows
window_overlap = int(window_size*(1-hop))
nfft = window_size
dimK = nfft//2+1

audio_file_length = 4.
audio_file_length_samples = int(audio_file_length * sr)
audio_file_offset = 1.
audio_file_offset_samples = int(audio_file_offset * sr)
af_start = audio_file_offset_samples
af_end = audio_file_offset_samples + audio_file_length_samples

rt60_decay = 0.05
nBands = 1
band_centerfreqs = np.empty(nBands)
band_centerfreqs[0] = 1000
for nb in range(1, nBands):
    band_centerfreqs[nb] = 2 * band_centerfreqs[nb - 1]

rt60_0s = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ir_length_seconds = rt60_0s[-1]
ir_length_samples = int(ir_length_seconds*sr)
rt_methods = ['edt', 't10', 't20', 't30']

# much faster, similar accuracy
Ls = [5]  # number of frames for the IIR filter

R = len(rt60_0s)
N = len(audio_files)
results = np.empty((R, N, len(Ls), dimM, len(rt_methods) ))
irs_estimated = np.empty((R, N, len(Ls), dimM, ir_length_samples))


# %% ROOM
h_t = 0
for rt60_0_idx, rt60_0 in enumerate(rt60_0s):
    print('--------------------------------------------')
    print('RT60', rt60_0)

    if rt60_0 == 0:
        h_t = np.asarray([1.]) # delta --> anechoic version
    else:
        room = np.array([10.2, 7.1, 3.2])
        rt60 = rt60_bands(rt60_0, nBands, rt60_decay)

        abs_wall = srs.find_abs_coeffs_from_rt(room, rt60)[0]

        # Critical distance for the room
        _, d_critical, _ = srs.room_stats(room, abs_wall, verbose=False)

        # Receiver position
        rec = (room / 2)[np.newaxis]  # center of the room
        nRec = rec.shape[0]

        # d_critical distance with defined angular position
        azi = np.random.rand() * 2 * np.pi
        incl = np.random.rand() * np.pi

        azi = azi + np.pi  # TODO: fix in srs library!!!
        src_sph = np.array([azi, np.pi / 2 - incl, d_critical.mean() / 2])
        src_cart = masp.sph2cart(src_sph)
        src = rec + src_cart
        nSrc = src.shape[0]

        # SH orders for receivers
        rec_orders = np.array([sh_order])

        maxlim = ir_length_seconds  # just stop if the echogram goes beyond that time ( or just set it to max(rt60) )
        limits = np.ones(nBands) * maxlim  # hardcoded!

        abs_echograms = srs.compute_echograms_sh(room, src, rec, abs_wall, limits, rec_orders)
        irs = srs.render_rirs_sh(abs_echograms, band_centerfreqs, sr).squeeze().T
        if irs.ndim == 1:
            irs = irs[np.newaxis, :]
        # Normalize as SN3D
        irs *= np.sqrt(4 * np.pi)

        # %% SYNTHESIZE AUDIOS

        for audio_file_idx, audio_file_path in enumerate(audio_files):
            print('FILE:', audio_file_path)

            s_t = librosa.core.load(audio_file_path, sr=sr, mono=True)[0][af_start:af_end]
            f, t, s_tf = scipy.signal.stft(s_t, sr, nperseg=window_size, noverlap=window_overlap,  nfft=nfft)

            # Get reverberant signal
            y_t = np.zeros((dimM, audio_file_length_samples))  # reverberant signal
            for m in range(dimM):
                y_t[m] = scipy.signal.fftconvolve(s_t, irs[m])[:audio_file_length_samples]  # keep original length

            # Add some noise... TODO

            f, t, y_tf = scipy.signal.stft(y_t, sr, nperseg=window_size, noverlap=window_overlap, nfft=nfft)
            dimM, dimK, dimN = y_tf.shape

            # %% MAR ANALYSIS

            # parameters
            p = 0.25
            i_max = 10
            ita = 1e-4
            epsilon = 1e-8

            tau = min(int(1 / hop), 1) # prevent it to be smaller than 1

            # norm_s_tf = np.linalg.norm(s_tf[0])
            # norm_y_tf = np.linalg.norm(y_tf[0])

            # estimate
            for L_idx, L in enumerate(Ls):
                est_s_tf, C, _ = estimate_MAR_sparse_parallel(y_tf, L, tau, p, i_max, ita, epsilon)
                _, est_s_t = scipy.signal.istft(est_s_tf, sr, nperseg=window_size,  noverlap=window_overlap, nfft=nfft)
                est_s_t = est_s_t[:, :audio_file_length_samples]

                # plot_magnitude_spectrogram(s_tf, sr)
                # plot_magnitude_spectrogram(y_tf, sr)
                # plot_magnitude_spectrogram(est_s_tf, sr)
                # plt.show()

                # coefs over frequency
                # plt.figure()
                # plt.plot(np.abs(C.squeeze()))
                # plt.show()


                # %% AKIS STFT SYSTEM IDENTIFICATION

                winsize = 8 * ir_length_samples # true, groundtruth
                hopsize = winsize / 16

                # IR ESTIMATION FROM ESTIMATED ANECHOIC SIGNAL
                for m in range(dimM):
                    ir_est_derv = sid_stft2(est_s_t[m], y_t[m], winsize, hopsize, ir_length_samples)
                    irs_estimated[rt60_0_idx, audio_file_idx, L_idx, m] = ir_est_derv


                    # %% t60 estimation

                    plot = False
                    est_derv = compute_t60(ir_est_derv, sr, band_centerfreqs, plot=plot, title='Estimated IR, derv signal')
                    est_derv = est_derv[0] # first band
                    # store
                    results[rt60_0_idx, audio_file_idx, L_idx, m] = est_derv


np.save('/Users/andres.perez/source/dereverberation/ctf/ctf_data/results_'+str(sh_order)+'_'+str(tau),results)
np.save('/Users/andres.perez/source/dereverberation/ctf/ctf_data/irs_estimated_'+str(sh_order)+'_'+str(tau),irs_estimated)


