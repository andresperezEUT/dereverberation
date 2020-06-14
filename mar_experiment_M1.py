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
dimM = 1  # first order

window_size = 512 # samples
hop = 1/2 # in terms of windows
window_overlap = window_size*(1-hop)
nfft = window_size
dimK = nfft//2+1

if hop == 1:
    window_type = 'boxcar'
else:
    window_type = 'hann'

audio_file_length = 10  ## seconds
audio_file_length_samples = int(audio_file_length * fs)

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

rt60_decay = 0.  # % of rt60_0 per octave

# rt60_vector = np.asarray([0.3, 0.5, 0.7, 0.9])
# rt60_vector = np.asarray([1.0])
rt60_vector = np.asarray([0.5])

# max_L =  int(np.ceil(0.5/(window_overlap/fs))) - tau
# max_L =  int(np.ceil(rt60_0rt60_0/(window_overlap/fs))) - tau


# filter_lengt_in_samples = 10 * window_size
# L = int(np.ceil(filter_lengt_in_samples / (window_size * hop)) ) #filter length in windows


L = 10
N = 10
I = 1

H = np.zeros((I, len(rt60_vector), nBands, N), dtype=complex)
# Hj = np.zeros((len(rt60_vector), nBands, N), dtype=complex)




for repetition in range(I):
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



        # %% SYNTHESIZE AUDIOS

        # af = audio_files[1]
        # af = '/Volumes/Dinge/audio/410298__inspectorj__voice-request-26b-algeria-will-rise-again-serious.wav'
        #
        # # Open audio files and encode into ambisonics

        # mono_s_t = librosa.core.load(af, sr=fs, mono=True)[0][:audio_file_length_samples]
        mono_s_t = np.random.normal(size=(audio_file_length_samples))

        s_t = mono_s_t
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
        ita = 1e-4
        epsilon = 1e-8

        norm_s_tf = np.linalg.norm(s_tf[0])
        norm_y_tf = np.linalg.norm(y_tf[0])

        # estimate
        est_s_tf, C, _ = estimate_MAR_sparse_parallel(y_tf, L, tau, p, i_max, ita, epsilon)
        # _, est_s_t = scipy.signal.istft(est_s_tf, fs, window=window_type, nperseg=window_size, noverlap=window_overlap, nfft=nfft)
        # est_s_t = est_s_t[:, :audio_file_length_samples]

        plot_magnitude_spectrogram(s_tf, fs)
        plot_magnitude_spectrogram(y_tf, fs)
        plot_magnitude_spectrogram(est_s_tf, fs)
        plt.show()

        # coefs over frequency
        plt.figure()
        plt.plot(np.abs(C.squeeze()))
        plt.show()



        # %%
        # stability


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


        band_centerfreqs_b = (band_centerfreqs / fs*2*(dimK-1)).astype(int)
        for nb, b in enumerate(band_centerfreqs_b):
            plt.figure()
            plt.suptitle('band' + str(band_centerfreqs[nb]))
            plt.imshow(np.abs(C[b])/np.max(np.abs(C[b])), cmap='magma')
            plt.grid()
            plt.colorbar()
            plt.figure()
            plt.suptitle('band' + str(band_centerfreqs[nb]))
            plt.imshow(np.abs(S[b])/np.max(np.abs(S[b])), cmap='magma')
            plt.grid()
            plt.colorbar()
            plt.show()


        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        e = np.max(np.abs(np.linalg.eigvals(S)), axis=1)
        for k in range(dimK):
            plt.plot(k, e[k], 'o-', color=colors[(e[k]>1).astype(int)])
        plt.show()



        # %%
        # impulse responses
        exp_lambda = lambda t, a, b: a * np.exp(-b * t)

        band_centerfreqs_b = (band_centerfreqs / fs*2*(dimK-1)).astype(int)
        for nb, b in enumerate(band_centerfreqs_b):

            # impulse at given freqneucies
            h_b = np.zeros(N, dtype='complex')
            for n in range(N):
                h_b[n] = np.power(S[b], n)[0,0]
            H[repetition, rt60_idx, nb] = h_b


            # # impulse avergaed over the flequencies
            # subbands_ks = J[nb]
            # num_subbands = subbands_ks.size
            # h_j = np.zeros((num_subbands, N, dimM * L, dimM * L), dtype='complex')
            # for k_idx, k in enumerate(subbands_ks):
            #     for n in range(1, N):
            #         h_j[k_idx, n] = np.power(S[k], n - 1)
            #     print(np.abs(h_j[k_idx].squeeze()))
            #     # h_j[k_idx] = np.zeros((num_subbands, N, dimM * L, dimM * L), dtype='complex')
            # h_j_mean = np.mean(np.abs(h_j), axis=0)
            # h_j_std = np.std(np.abs(h_j), axis=0)
            # print(h_j_mean.squeeze())
            # Hj[rt60_idx, nb] = h_j_mean


            plt.figure()
            # plt.yscale('log')
            plt.suptitle('band' + str(band_centerfreqs[nb]))

            plt.plot(np.arange(N), np.abs(h_b), label='fine')
            # plt.errorbar(np.arange(1, N), h_j_mean[1:, m1, m2], yerr=h_j_std[1:, m1, m2],  label='mean(abs)',)
            plt.plot(np.arange(N), exp_lambda(np.arange(N) * hop * window_size, 1, decay_rate(rt60, fs)[nb]), label='theoretical')

            plt.legend()
            plt.show()


        # impulse at subbands (mean of impulses)
        # start_k = J[nb][0]
        # end_k = J[nb][-1]+1
        # C_j = np.mean(C[start_k:end_k], axis=0)

        # num_k = J[nb].size
        # h_j = np.zeros((num_k, N, dimM, dimM), dtype='complex')
        # for k_idx, k, in enumerate(J[nb]): # all ks for the given subband
        #
        #     for n in range(1, N):
        #         h_j[k_idx, n] = np.power(C[k], n-1)
        #
        # Hj[rt60_idx, nb] = np.mean(np.abs(h_j), axis=0)
        #
        # plt.figure()
        # plt.title('rt60='+str(rt60_0) + ' nb'+str(nb))
        # plt.plot(range(1,N), np.mean(np.abs(h_b)[1:], axis=(-2, -1)), label='freq')
        # plt.errorbar(range(1,N), np.mean(np.mean(np.abs(h_j), axis=0)[1:], axis=(-2, -1)),
        #              np.mean(np.std(np.abs(h_j), axis=0)[1:], axis=(-2, -1)))
        # plt.show()

# %%
# plot


# # for o in range(dimM):
# for nb in range(nBands):
#     plt.figure()
#     # plt.yscale('log')
#     plt.suptitle('band'+str(band_centerfreqs[nb]))
#
#     for rt60_idx, rt60_0 in enumerate(rt60_vector):
#         # plt.plot(range(1, N), np.mean(np.abs(H[rt60_idx, nb, 1:N, :, :]), axis=(-2,-1)), label=rt60_0)
#
#         h = H[rt60_idx, nb, 1:N, 0, 0]
#         plt.plot(range(1, N), np.abs(h), label=rt60_0)
#
#     plt.legend()
#     plt.show()
#
#     # plt.figure()
#     # plt.yscale('log')
#     # plt.suptitle('band '+str(band_centerfreqs[nb]))
#     #
#     # for rt60_idx, rt60_0 in enumerate(rt60_vector):
#     #     plt.plot(range(1, display_N), np.mean(np.abs(Hj[rt60_idx, nb, 1:display_N, :, :]), axis=(-2,-1)), label='rt60='+str(rt60_0))
#     #
#     # plt.xlabel('L')
#     # plt.ylim(1e-2, 1e0)
#     # plt.legend()
#     # plt.show()
#
