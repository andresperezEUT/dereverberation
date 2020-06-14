import numpy as np
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



# %% PARAMETERS

#### parameters from "Online Dereverberation..."

fs = 16000
dimM = 4  # first order

window_size = 512 # samples
window_overlap = window_size//2
# window_overlap = 0
nfft = window_size
window_type = 'hann'

audio_file_length = 4  ## seconds

## MAR-------
# D = int(np.floor(ir_start_time * fs / window_overlap)) # ir start frame
tau = window_size//window_overlap
print('tau=',tau)
if tau < 1:
    raise Warning('D should be at least 1!!!')
# L = 10  # total number of frames for the filter

alpha = 0.4
ita = np.power(10, -35 / 10)
alpha_RLS = 0.99

noise_power = 1e-5



# get audio files
audio_files = []
data_folder_path = '/Volumes/Dinge/DSD100subset/Sources'
for root, dir, files in os.walk(data_folder_path):
    for f in files:
        extension = os.path.splitext(f)[1]
        if 'wav' in extension:
            audio_files.append(os.path.join(root, f))



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

# rt60_vector = np.arange(0.3,1,0.15)
rt60_vector = np.arange(0.3,1,0.3)
# L_vector = np.arange(3, 19, 3)
L_vector = np.arange(3, 19, 6)

B = np.empty((rt60_vector.size, L_vector.size, dimM, dimM))

for rt60_idx, rt60 in enumerate(rt60_vector):

    room = np.array([10.2, 7.1, 3.2])
    rt60 = np.array([rt60])
    nBands = len(rt60)

    # Generate octave bands
    band_centerfreqs = np.empty(nBands)
    # band_centerfreqs[0] = 125
    # for nb in range(1, nBands):
    #     band_centerfreqs[nb] = 2 * band_centerfreqs[nb - 1]
    band_centerfreqs[0] = 1000

    abs_wall = srs.find_abs_coeffs_from_rt(room, rt60)[0]

    # Critical distance for the room
    _, d_critical, _ = srs.room_stats(room, abs_wall, verbose=False)

    # Receiver position
    rec = (room/2)[np.newaxis] # center of the room
    nRec = rec.shape[0]

    # d_critical distance with defined angular position
    azi = azi + np.pi # TODO: fix in srs library!!!
    src_sph = np.array([azi, np.pi/2-incl, d_critical.mean()])
    src_cart = masp.sph2cart(src_sph)
    src = rec + src_cart
    nSrc = src.shape[0]

    # SH orders for receivers
    rec_orders = np.array([1])

    maxlim = 0.5  # just stop if the echogram goes beyond that time ( or just set it to max(rt60) )
    # limits = np.minimum(rt60, maxlim)
    limits = np.asarray([maxlim])  # hardcoded!

    abs_echograms = srs.compute_echograms_sh(room, src, rec, abs_wall, limits, rec_orders)
    irs = srs.render_rirs_sh(abs_echograms, band_centerfreqs, fs).squeeze().T
    # Normalize as SN3D
    irs *= np.sqrt(4 * np.pi)
    irs *= np.asarray([1, 1. / np.sqrt(3), 1. / np.sqrt(3), 1. / np.sqrt(3)])[:,np.newaxis]  ## ACN, SN3D




# %% SYNTHESIZE AUDIOS

    # # af = audio_files[1]
    af = '/Volumes/Dinge/audio/410298__inspectorj__voice-request-26b-algeria-will-rise-again-serious.wav'
    #
    # # Open audio files and encode into ambisonics
    audio_file_length_samples = int(audio_file_length * fs)

    mono_s_t = librosa.core.load(af, sr=fs, mono=True)[0][audio_file_length_samples:2*audio_file_length_samples]
    # mono_s_t = np.random.normal(size=(audio_file_length_samples))


    s_t = mono_s_t * y.T  # dry ambisonic target
    f, t, s_tf = scipy.signal.stft(s_t, fs, window=window_type, nperseg=window_size, noverlap=window_overlap, nfft=nfft)

    # Get reverberant signal
    y_t = np.zeros((dimM, audio_file_length_samples))  # reverberant signal
    for m in range(dimM):
        y_t[m] = scipy.signal.fftconvolve(mono_s_t, irs[m])[:audio_file_length_samples]  # keep original length

    # Add some noise...
    # y_t += np.random.normal(scale=noise_power, size=y_t.size).reshape(y_t.shape)

    f, t, y_tf = scipy.signal.stft(y_t, fs, window=window_type, nperseg=window_size, noverlap=window_overlap, nfft=nfft)
    dimM, dimK, dimN = y_tf.shape


# %% ANALYSIS

    # parameters
    p = 0.25
    i_max = 20
    ita = 0.03
    epsilon = 1e-8

    norm_s_tf = np.linalg.norm(s_tf[0])
    norm_y_tf = np.linalg.norm(y_tf[0])
    norm_est_s_tf = np.empty(L_vector.size)
    norm_diff = np.empty(L_vector.size)

    est_s_tf_l = np.zeros((L_vector.size, dimM, dimK, dimN), dtype='complex')
    est_s_t_l = np.zeros((L_vector.size, dimM, audio_file_length_samples))
    C_l = []

    # compute
    for l_idx, L in enumerate(L_vector):


        # # plot ir
        # ir = irs[0]
        # n = np.arange(ir.size)
        # a = decay_rate(rt60, fs)
        # decay = np.exp(-a * n)
        #
        # plt.figure()
        # plt.suptitle('rt60=' + str(rt60) + " L=" + str(L))
        # plt.plot(n, np.abs(ir))
        # plt.plot(n, decay)
        # plt.vlines(tau * window_overlap, 0.1, 1, 'r')
        # plt.vlines((tau + L) * window_overlap, 0.1, 1, 'r')
        # plt.show()


        # estimate
        est_s_tf, C, _ = estimate_MAR_sparse_parallel(y_tf, L, tau, p, i_max, ita, epsilon)
        est_s_tf_l[l_idx] = est_s_tf
        C_l.append(C)

        norm_est_s_tf[l_idx] = np.linalg.norm(est_s_tf[0])

        abs_diff = np.abs(s_tf) - np.abs(est_s_tf)
        norm_diff[l_idx] = np.linalg.norm(abs_diff[0])


# %%
# istft
    for l_idx, L in enumerate(L_vector):

        est_s_tf = est_s_tf_l[l_idx]
        _, est_s_t = scipy.signal.istft(est_s_tf, fs, window=window_type, nperseg=window_size, noverlap=window_overlap,
                                        nfft=nfft)
        est_s_t = est_s_t[:, :audio_file_length_samples]
        est_s_t_l[l_idx] = est_s_t



# %%
# resynthesis
    for l_idx, L in enumerate(L_vector):

        est_s_tf = est_s_tf_l[l_idx]
        C = C_l[l_idx]

        x2_tf = np.zeros((dimM, dimK, dimN), dtype=complex)
        for k in range(dimK):
            for n in range(tau, dimN):

                x = np.zeros(L * dimM, dtype=complex)
                for m in range(dimM):
                    for l in range(L):
                        x[L * m + l] = x2_tf[m, k, n - tau - l]

                x2_tf[:, k, n] = est_s_tf[:, k, n] + x @ C[k, :, :]

        plot_magnitude_spectrogram(x2_tf, 'x2_tf')
        # difference
        plot_magnitude_spectrogram(np.abs(y_tf) - np.abs(x2_tf), 'abs diff')
        plt.show()


# %%
# stability
    for l_idx, L in enumerate(L_vector):

        C = C_l[l_idx]
        # Build square transition matrix
        S = np.zeros((dimK, dimM*L, dimM*L), dtype=complex)
        for k in range(dimK):
            for l in range(L):
                for m1 in range(dimM):
                    for m2 in range(dimM):
                        S[k, m1, l*dimM+m2] = C[k, m1*L+l, m2]
                        # print(m1, m2, l, '    ', m1, l*dimM+m2, '     ', m1*L+l, m2)
            # add I(ML)
            S[k, dimM:, :] = np.identity(dimM*L)[:-dimM, :]

        plt.figure()
        plt.imshow(np.abs(C[k]), cmap='magma')
        plt.grid()
        plt.colorbar()
        plt.figure()
        plt.imshow(np.abs(S[k]), cmap='magma')
        plt.grid()
        plt.colorbar()
        plt.show()


        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        e = np.max(np.abs(np.linalg.eigvals(S)), axis=1)
        for k in range(dimK):
            plt.plot(k, e[k], 'o-', color=colors[(e[k]>1).astype(int)])
        plt.show()


# # %%
#     # decays
#     colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # default colormap
#     exp_lambda = lambda t, a, b: a * np.exp(-b * t)
#
#     for l_idx, L in enumerate(L_vector):
#         fig, axs = plt.subplots(1, dimM, sharey=True)
#         fig.subplots_adjust(wspace=0)
#
#         C = C_l[l_idx]
#         mean_C = np.mean(np.abs(C),axis=0)
#
#         vmax = np.max(mean_C)
#         vmin = np.min(mean_C)
#         axs[0].set_ylabel(r'$\bar{g}_{mi}(\ell)$')
#
#         # m1 is output channel
#         for m1 in range(dimM):
#             axs[m1].set_title('Output i='+str(m1))
#             axs[m1].set_ylim((vmin, vmax))
#             axs[m1].set_xlabel('$\ell$')
#
#             # m2 is input channel
#             for m2 in range(dimM):
#                 data = mean_C[m2 * L:(m2 + 1) * L, m1]
#                 axs[m1].plot(data, label='Input m='+str(m2), c=colors[m2])
#                 a, b = scipy.optimize.curve_fit(exp_lambda, np.arange(L), data)[0]
#                 B[rt60_idx, l_idx, m1, m2] = b / window_overlap  # compensate for the frame length
#                 axs[m1].plot(exp_lambda(np.arange(L), a, b), ls='--', c=colors[m2])
#
#         plt.yscale('log')
#         plt.legend()
#         plt.show()
#
#
#
# # Mean across all components
# colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] # default colormap
#
#
# # average over all
# plt.figure()
# for rt60_idx, rt60 in enumerate(rt60_vector):
#     plt.errorbar(L_vector, np.mean(B[rt60_idx], axis=(1,2))*window_overlap, yerr=np.std(B[rt60_idx], axis=(1,2))*window_overlap, fmt='o-', label='rt60='+str(rt60))
#     plt.hlines(decay_rate(rt60, fs)*window_overlap, L_vector[0], L_vector[-1], linestyles='--', color=colors[rt60_idx])
#     plt.xlabel('$\ell$')
#     plt.ylabel(r'$ mean( \alpha_{mi}(\ell) )$')
#     plt.legend()
# plt.show()
#
# # # average over same output
# # plt.figure()
# # plt.suptitle('exponential L')
# # for rt60_idx, rt60 in enumerate(rt60_vector):
# #     plt.errorbar(L_vector, np.mean(B[rt60_idx, :, :, 0], axis=(-1))*window_overlap, yerr=np.std(B[rt60_idx, :, :, 0]*window_overlap, axis=(-1)), fmt='o-', label='rt60='+str(rt60))
# #     plt.hlines(decay_rate(rt60, fs)*window_overlap, L_vector[0], L_vector[-1], linestyles='--', color=colors[rt60_idx])
# #     plt.xlabel('$\ell$')
# #     plt.ylabel(r'$\alpha$')
# #     plt.legend()
# # plt.show()
#
# # Only with (0,0)
# # IN FRAMES
# plt.figure()
# for rt60_idx, rt60 in enumerate(rt60_vector):
#     plt.errorbar(L_vector, B[rt60_idx,:, 0, 0]*window_overlap, fmt='o-', c=colors[rt60_idx], label='rt60='+str(rt60))
#     plt.hlines(decay_rate(rt60, fs)*window_overlap,  L_vector[0], L_vector[-1], linestyles='--', color=colors[rt60_idx])
#     plt.xlabel('$\ell$')
#     plt.ylabel(r'$\alpha_{00}(\ell)$')
#     plt.legend()
# plt.show()
#
# # # IN SAMPLES
# # plt.figure()
# # plt.suptitle('exponential L')
# # for rt60_idx, rt60 in enumerate(rt60_vector):
# #     plt.errorbar(L_vector*window_overlap, B[rt60_idx,:, 0, 0], fmt='o-', c=colors[rt60_idx], label='rt60='+str(rt60))
# #     plt.hlines(decay_rate(rt60, fs),  L_vector[0]*window_overlap, L_vector[-1]*window_overlap, linestyles='--', color=colors[rt60_idx])
# #     plt.xlabel('$\ell$')
# #     plt.ylabel(r'$\alpha$')
# #     plt.legend()
# # plt.show()
#
