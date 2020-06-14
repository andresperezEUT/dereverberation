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
hop = 1/2 # in terms of windows
window_overlap = window_size*(1-hop)
nfft = window_size
dimK = nfft//2+1

if hop == 1:
    window_type = 'boxcar'
else:
    window_type = 'hann'

audio_file_length = 20  ## seconds

## MAR-------
# D = int(np.floor(ir_start_time * fs / window_overlap)) # ir start frame
tau = int(1/hop)
print('tau=',tau)
if tau < 1:
    raise Warning('D should be at least 1!!!')

noise_power = 1e-5



# get audio files
audio_files = []
data_folder_path = '/Volumes/Dinge/DSD100subset/Sources'
for root, dir, files in os.walk(data_folder_path):
    for f in files:
        extension = os.path.splitext(f)[1]
        if 'wav' in extension:
            audio_files.append(os.path.join(root, f))




# %%
# FREQUENCY BANDS
freqs = np.arange(dimK)*fs/2/(dimK-1) # frequency indices to Hz

nBands = 5
band_centerfreqs = np.empty(nBands)
band_centerfreqs[0] = 250
for nb in range(1, nBands):
    band_centerfreqs[nb] = 2 * band_centerfreqs[nb - 1]

# Octave frequency bands
J = []
for nb, f0 in enumerate(band_centerfreqs):
    l = f0 / np.sqrt(2)
    r = f0 * np.sqrt(2)
    f_range = freqs[np.logical_and(l <= freqs, freqs < r)]
    k_range = f_range / fs*2*(dimK-1)
    J.append(k_range.astype(int))



# k_ranges = []
# for nb in range(nBands):
#     if nb < nBands-1:
#         k_indices_inside_range = freqs[np.logical_and(band_centerfreqs[nb] <= freqs, freqs < band_centerfreqs[nb+1])] / fs*2*(dimK-1) # Hz to frequency indices
#     else: # last
#         k_indices_inside_range = freqs[band_centerfreqs[nb] <= freqs] / fs*2*(dimK-1)
#     k_ranges.append(k_indices_inside_range.astype(int))
#     # print(band_centerfreqs[nb], k_ranges[nb])



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


def rt60_bands(rt60_0, nnBands, decay=0.1):
    # decay per octave
    return np.asarray([rt60_0-(rt60_0*decay*i) for i in range(nBands)])

rt60_decay = 0.1  # per octave

rt60_vector = np.asarray([0.5])
# rt60_vector = np.asarray([0.5])

# max_L =  int(np.ceil(0.5/(window_overlap/fs))) - tau
# max_L =  int(np.ceil(rt60_0rt60_0/(window_overlap/fs))) - tau


# filter_lengt_in_samples = 10 * window_size
# L = int(np.ceil(filter_lengt_in_samples / (window_size * hop)) ) #filter length in windows


# L_vector = np.arange(3, max_L, 5)
# L_vector = np.asarray([L])
L_vector = np.asarray([2,3,4,5])


B = np.empty((nBands, rt60_vector.size, L_vector.size, dimM, dimM))
# B_std = np.empty((nBands, rt60_vector.size, L_vector.size, dimM, dimM))
B_diff = np.empty((nBands, rt60_vector.size, L_vector.size, dimM, dimM))

for rt60_idx, rt60_0 in enumerate(rt60_vector):

    room = np.array([10.2, 7.1, 3.2])
    rt60 = rt60_bands(rt60_0, nBands, rt60_decay)

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

    maxlim = rt60_0  # just stop if the echogram goes beyond that time ( or just set it to max(rt60) )
    limits = np.ones(nBands)*maxlim # hardcoded!

    abs_echograms = srs.compute_echograms_sh(room, src, rec, abs_wall, limits, rec_orders)
    irs = srs.render_rirs_sh(abs_echograms, band_centerfreqs, fs).squeeze().T
    # Normalize as SN3D
    irs *= np.sqrt(4 * np.pi)
    irs *= np.asarray([1, 1. / np.sqrt(3), 1. / np.sqrt(3), 1. / np.sqrt(3)])[:,np.newaxis]  ## ACN, SN3D




# %% SYNTHESIZE AUDIOS

    af = audio_files[1]
    # af = '/Volumes/Dinge/audio/410298__inspectorj__voice-request-26b-algeria-will-rise-again-serious.wav'
    #
    # # Open audio files and encode into ambisonics
    audio_file_length_samples = int(audio_file_length * fs)

    mono_s_t = librosa.core.load(af, sr=fs, mono=True)[0][:audio_file_length_samples]
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

    est_s_tf_l = np.zeros((L_vector.size, dimM, dimK, dimN), dtype='complex')
    est_s_t_l = np.zeros((L_vector.size, dimM, audio_file_length_samples))
    C_l = []

    # compute
    for l_idx, L in enumerate(L_vector):

        # estimate
        est_s_tf, C, _ = estimate_MAR_sparse_parallel(y_tf, L, tau, p, i_max, ita, epsilon)
        est_s_tf_l[l_idx] = est_s_tf
        C_l.append(C)




# %%
    # decays
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']*2  # default colormap
    exp_lambda = lambda t, a, b: a * np.exp(-b * t)


    for nb in range(nBands):

        ks = J[nb]

        for l_idx, L in enumerate(L_vector):
            fig, axs = plt.subplots(1, dimM, sharey=True)
            fig.subplots_adjust(wspace=0)
            plt.yscale('log')

            C = C_l[l_idx]  # [dimK, dimM * L, dimM]
            mean_C = np.mean(np.abs(C[ks[0]:ks[-1]+1]),axis=0)

            vmax = np.max(mean_C)
            vmin = np.min(mean_C)
            axs[0].set_ylabel(r'$\bar{g}_{mi}(\ell), $'+'L='+str(L)+", RT60_0="+str(rt60_0)+', B='+str(nb))

            # m1 is output channel
            for m1 in range(dimM):
                axs[m1].set_title('Output i='+str(m1))
                axs[m1].set_ylim((vmin, vmax))
                axs[m1].set_xlabel('$\ell$')

                # m2 is input channel
                for m2 in range(dimM):

                    data = mean_C[m2 * L:(m2 + 1) * L, m1]
                    (a, b), pcov = scipy.optimize.curve_fit(exp_lambda, np.arange(L), data)
                    fit_data = exp_lambda(np.arange(L), a, b)

                    axs[m1].plot(data, label='Input m='+str(m2), c=colors[m2])
                    axs[m1].plot(fit_data, ls='--', c=colors[m2])

                    B[nb, rt60_idx, l_idx, m1, m2] = b * window_size * hop  # IN SAMPLES!!
                    B_diff[nb, rt60_idx, l_idx, m1, m2] = np.linalg.norm(data - fit_data)

            plt.legend()
            plt.show()



# # RT60 -  Mean of all components
# plt.figure()
# for rt60_idx, rt60_0 in enumerate(rt60_vector):
#
#     alpha = B[:,rt60_idx, l_idx]*window_overlap
#
#     t = get_rt60(alpha, fs)
#     mean_t =np.mean(t, axis=(1,2))
#     std_t = np.std(t, axis=(1,2))
#
#     plt.errorbar(band_centerfreqs, mean_t, yerr=std_t, fmt='o-', color=colors[rt60_idx])
#     # true alpha
#     rt60 = rt60_bands(rt60_0, nBands, rt60_decay)
#     plt.errorbar(band_centerfreqs, rt60, fmt='^', ls='--', color=colors[rt60_idx])
#
#     plt.xlabel('bands (Hz)')
#     plt.ylabel('RT60')
#     plt.xscale('log')
# plt.show()


# ALPHA -  Mean of all components
# for l_idx, L in enumerate(L_vector):
#
#     plt.figure()
#     plt.ylim(0, 0.003)
#     for rt60_idx, rt60_0 in enumerate(rt60_vector):
#
#         alpha = B[:, rt60_idx, l_idx]
#         mean_alpha = np.mean(alpha, axis=(1,2))
#         std_alpha = np.std(alpha, axis=(1,2))
#
#         plt.errorbar(band_centerfreqs, mean_alpha, yerr=std_alpha, fmt='o-', color=colors[rt60_idx])
#
#         # true alpha
#         rt60 = rt60_bands(rt60_0, nBands, rt60_decay)
#         plt.errorbar(band_centerfreqs,  decay_rate(rt60, fs), fmt='^', ls='--', color=colors[rt60_idx], label='RT60_0='+str(rt60_0))
#
#         plt.xlabel('bands (Hz)')
#         plt.ylabel(r'$\alpha, L=$'+str(L))
#         plt.xscale('log')
#         plt.legend()
#     plt.show()


# mean of all components - different L, same rt60
for rt60_idx, rt60_0 in enumerate(rt60_vector):
    plt.figure()
    # plt.ylim(0, 0.8)

    # true alpha
    rt60 = rt60_bands(rt60_0, nBands, rt60_decay)
    plt.errorbar(band_centerfreqs, decay_rate(rt60, fs), fmt='^', ls='--', color=colors[len(L_vector)],
                 label='RT60_0=' + str(rt60_0))

    for l_idx, L in enumerate(L_vector):

        alpha = B[:, rt60_idx, l_idx]
        mean_alpha = np.mean(alpha, axis=(1,2))
        std_alpha = np.std(alpha, axis=(1,2))

        plt.errorbar(band_centerfreqs, mean_alpha, yerr=std_alpha, fmt='o-', color=colors[l_idx], label='L='+str(L))

        plt.xlabel('bands (Hz)')
        plt.ylabel(r'$\alpha, RT60=$'+str(rt60_0))
        plt.xscale('log')
        plt.legend()
    plt.show()
#


# %%


### curve fit error

# B_diff [nBands, rt60_vector.size, L_vector.size, dimM, dimM]

plt.plot(L_vector, B_diff[-1,-1,:,0,0])
plt.show()

# Plot magnitude spectrograms
plot_magnitude_spectrogram(s_tf[0], title='true')
for l_idx, L in enumerate(L_vector):
    est_s_tf = est_s_tf_l[l_idx]
    plot_magnitude_spectrogram(est_s_tf[0], title=L)
    plt.show()

# Get istfts
est_s_t_l = []
for l_idx, L in enumerate(L_vector):
    est_s_tf = est_s_tf_l[l_idx]
    _, est_s_t = scipy.signal.istft(est_s_tf, fs, window=window_type, nperseg=window_size, noverlap=window_overlap,
                                nfft=nfft)
    est_s_t = est_s_t[:, :audio_file_length_samples]
    est_s_t_l.append(est_s_t)
#
# play(s_t[0,:5*fs], fs)
# play(y_t[0,:5*fs], fs)
# for l_idx, L in enumerate(L_vector):
#     play(est_s_t_l[l_idx][0,:5*fs], fs)




# %%
# resynthesis


C = C_l[0]

dimM, dimK, dimN = y_tf.shape

delta = np.zeros((4, fs))
delta[:,fs//2] = 1.

f, t, s_tf = scipy.signal.stft(s_t, fs, window=window_type, nperseg=window_size, noverlap=window_overlap, nfft=nfft)

_, _, delta_tf = scipy.signal.stft(delta, fs, window=window_type, nperseg=window_size, noverlap=window_overlap, nfft=nfft)
dimM, dimK, dimN = delta_tf.shape


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

            x2_tf[:, k, n] = delta_tf[:, k, n] + x @ C[k, :, :]

    plot_magnitude_spectrogram(delta_tf, 'delta_tf')
    plot_magnitude_spectrogram(x2_tf, 'x2_tf')
    # difference
    # plot_magnitude_spectrogram(np.abs(y_tff) - np.abs(x2_tf), 'abs diff')
    plt.show()




# %%
# stability
    for l_idx, L in enumerate(L_vector):
    # L = L_vector[-1]
    # l_idx = 3

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
