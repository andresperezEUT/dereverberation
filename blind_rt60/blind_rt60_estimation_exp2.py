"""
esperiment 2

test with a range of rt60s, different sources
"""

import numpy as np
import librosa
import scipy.signal
import matplotlib.pyplot as plt
from methods import plot_magnitude_spectrogram

pp = "/Users/andres.perez/source/masp"
import sys
sys.path.append(pp)
import masp
from masp import shoebox_room_sim as srs
import os
import scipy.stats
from blind_rt60.datasets import get_audio_files
from blind_rt60.blind_rt60_methods import *


plt.style.use('seaborn-whitegrid')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # default colormap



# %% 0. RESULTS

# rt60 0.4 - 0.7
# ENST - ALL :  m, n (4.7725316722881805, -0.7889018101833457)
# DSD100 - all (drums) : m, n = (3.4234199209082963, -0.45679586527031363)
# IDMT - ALL (DRUMS) M, N (3.3736679826651144, -0.34783194520880767)

# dsd 100 valid indices:
# valid_audio_file_idx = np.array([ 0,  1,  2,  4,  5,  6,  8,  9, 11, 12, 15, 16, 18, 19, 20, 22, 23,
#        24, 29, 31, 32, 33, 34, 35, 37, 38, 39, 41, 42, 43, 44, 46, 47, 48,
#        49, 51, 52, 53, 54, 56, 57, 58, 59, 60, 63, 65, 67, 68, 69, 72, 74,
#        77, 79, 80, 81, 83, 84, 85, 86, 87, 88, 89, 91, 92, 94, 95, 96, 98])


# rt60 0.4 - 1
# dsd100 - all (drums): m, n = (6.123920485012435, -1.2566895887421592)

# %% 0. Prepare data

# get audio files
num_files = -1
dataset = 'DSD100'
subset = ''
main_path = '/Volumes/Dinge/datasets'
audio_files = get_audio_files(main_path, dataset, subset)[:num_files]

sr = 8000 # target sample rate
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

sh_order = 0
dimM = (sh_order + 1) ** 2

rt60_0s = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # ANECHOIC MUST BE FIRST!!!

R = len(rt60_0s)
N = len(audio_files)
median_RT = np.zeros((R, N))


# %% ROOM
h_t = 0
for rt60_0_idx, rt60_0 in enumerate(rt60_0s):

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

        maxlim = rt60_0  # just stop if the echogram goes beyond that time ( or just set it to max(rt60) )
        limits = np.ones(nBands) * maxlim  # hardcoded!

        abs_echograms = srs.compute_echograms_sh(room, src, rec, abs_wall, limits, rec_orders)
        irs = srs.render_rirs_sh(abs_echograms, band_centerfreqs, sr).squeeze().T
        if irs.ndim == 1:
            irs = irs[np.newaxis, :]
        # Normalize as SN3D
        irs *= np.sqrt(4 * np.pi)
        h_t = irs[0]

    # %% 0. Iterate

    for audio_file_idx, audio_file_path in enumerate(audio_files):
        s_t = librosa.core.load(audio_file_path, sr=sr, mono=True)[0][af_start:af_end]

        print('--------------------------------------------')
        print('ITER:', audio_file_idx)
        print('FILE:', audio_file_path)

        # TODO
        r_t = scipy.signal.fftconvolve(s_t, h_t)[:audio_file_length_samples]  # keep original length

        window_size = 1024
        window_overlap = window_size // 4
        nfft = window_size
        f, t, r_tf = scipy.signal.stft(r_t, sr, nperseg=window_size, noverlap=window_overlap, nfft=nfft)

        FDR_time_limit = 0.5
        try:
            median_RT[rt60_0_idx, audio_file_idx] = estimate_blind_rt60(r_tf, sr, window_overlap, FDR_time_limit)
        except ValueError:
            median_RT[rt60_0_idx, audio_file_idx] = 0


# %%
# %%
# %%
# %%
# %% ANALYSIS


results = median_RT.T

results = np.load('/Users/andres.perez/source/dereverberation/blind_rt60/blind_rt60_data/results_.npy')
#
# results[:, 0] = np.load('blind_rt60_data/drummer2_dry.npy')
# results[:, 1] = np.load('blind_rt60_data/drummer2_300.npy')
# results[:, 2] = np.load('blind_rt60_data/drummer2_500.npy')
# results[:, 3] = np.load('blind_rt60_data/drummer2_700.npy')

# RT60 computed
plt.figure()
plt.suptitle(dataset + " - " + subset)
plt.subplot(321)
plt.title('Computed RT60')
plt.plot(results[:,1:], '-o', markersize=2)
plt.plot(results[:,0], '--', markersize=2)

# Difference wrt dry
diff = results[:,1:] - results[:,0,np.newaxis]
plt.subplot(322)
plt.title('Computed RT60 - Difference')
plt.plot(diff)

# filter it
th = 0.02
mask = diff>th

# indices of recordings passing the test,
# and results for those recordings
masked_indices = []
masked_results = []
for r in range(R-1):
    iii = np.argwhere(mask[:,r]).squeeze()
    rrr = results[:, r+1][iii]
    masked_indices.append(iii)
    masked_results.append(rrr)

# maximum length, for visualization
maxN = 0
for r in range(R-1):
    l = len(masked_indices[r])
    if l > maxN:
        maxN = l

# Plot
plt.subplot(323)
plt.title('masked results, th='+str(th))
for r in range(R-1):
    L = len(masked_results[r])
    plt.plot(masked_results[r], '-o', markersize=2, label=rt60_0s[r+1], color=colors[r])
    plt.hlines(rt60_0s[r+1], 0, maxN-1, linestyle='--', color=colors[r])
plt.legend()

# Statistics
masked_indices = []
masked_results = []
for r in range(R-1):
    iii = np.argwhere(mask[:,r]).squeeze()
    rrr = results[:, r+1][iii]
    masked_indices.append(iii)
    masked_results.append(rrr)

mean_values = []
std_values = []
for r in range(R-1):
    mean_values.append(np.mean(masked_results[r]))
    std_values.append(np.std(masked_results[r]))

plt.subplot(324)
plt.title('Masked results - stats & regression')
plt.errorbar(rt60_0s[1:], mean_values, yerr=std_values)
plt.plot(rt60_0s[1:], rt60_0s[1:], linestyle='--')

# m, n, r_value, p_value, std_err = scipy.stats.linregress(mean_values, rt60_0s[1:])

x = np.asarray(rt60_0s[1:])
y = np.asarray(mean_values)
p0 = 2, 1 # initial guess
popt, pcov = scipy.optimize.curve_fit(line, y, x, p0, sigma=std_values, absolute_sigma=True)
yfit = line(y, *popt)
m, n = popt
plt.plot(rt60_0s[1:], np.asarray(mean_values)*m+n, '-o', markersize=2)


# save
np.save('/Users/andres.perez/source/dereverberation/blind_rt60/blind_rt60_data/results_'+subset,results)
np.save('/Users/andres.perez/source/dereverberation/blind_rt60/blind_rt60_data/masked_results_'+subset,masked_results)
# %% Test!!

# Take a different excerpt of the recorded audios and measure the error

audio_file_offset = 5.
audio_file_offset_samples = int(audio_file_offset * sr)
af_start = audio_file_offset_samples
af_end = audio_file_offset_samples + audio_file_length_samples


median_RT_eval = [ [] for r in range(len(rt60_0s[1:])) ]


for rt60_0_idx, rt60_0 in enumerate(rt60_0s[1:]): #Excluding anechoic case
    h_t = 0
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

        maxlim = rt60_0  # just stop if the echogram goes beyond that time ( or just set it to max(rt60) )
        limits = np.ones(nBands) * maxlim  # hardcoded!

        abs_echograms = srs.compute_echograms_sh(room, src, rec, abs_wall, limits, rec_orders)
        irs = srs.render_rirs_sh(abs_echograms, band_centerfreqs, sr).squeeze().T
        if irs.ndim == 1:
            irs = irs[np.newaxis, :]
        # Normalize as SN3D
        irs *= np.sqrt(4 * np.pi)
        h_t = irs[0]

    for audio_file_idx, audio_file_path in enumerate(audio_files):

       # Analyse only the masked audios

       if audio_file_idx in masked_indices[rt60_0_idx]:

            s_t = librosa.core.load(audio_file_path, sr=sr, mono=True)[0][af_start:af_end]

            print('--------------------------------------------')
            print('ITER:', audio_file_idx)
            print('FILE:', audio_file_path)

            # TODO
            r_t = scipy.signal.fftconvolve(s_t, h_t)[:audio_file_length_samples]  # keep original length

            window_size = 1024
            window_overlap = window_size // 4
            nfft = window_size
            f, t, r_tf = scipy.signal.stft(r_t, sr, nperseg=window_size, noverlap=window_overlap, nfft=nfft)

            FDR_time_limit = 0.5
            try:
                res = estimate_blind_rt60(r_tf, sr, window_overlap, FDR_time_limit)
                median_RT_eval[rt60_0_idx].append(res)
            except ValueError:
                pass # just don't add it to our results



np.save('/Users/andres.perez/source/dereverberation/blind_rt60/blind_rt60_data/median_RT_eval_'+subset,median_RT_eval)

# %%

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
median_RT_eval = np.load('/Users/andres.perez/source/dereverberation/blind_rt60/blind_rt60_data/median_RT_eval_.npy')
np.load = np_load_old



m, n = (6.123920485012435, -1.2566895887421592)


#    plot final results
plt.subplot(325)
plt.title('Eval results')
for rt60_0_idx, rt60_0 in enumerate(rt60_0s[1:]):  # Excluding anechoic case
    N = len(median_RT_eval[rt60_0_idx])
    # plt.plot(np.asarray(median_RT_eval[rt60_0_idx]), color=colors[rt60_0_idx])
    plt.plot(np.asarray(median_RT_eval[rt60_0_idx])*m+n, '-o', markersize=2, color=colors[rt60_0_idx], label=str(rt60_0))
    plt.hlines(rt60_0, 0, N-1, linestyle='--', color=colors[rt60_0_idx])
plt.legend()

# mean squared error

mse = np.zeros(R-1)
mean_error = np.zeros(R-1)
std_error = np.zeros(R-1)
for rt60_0_idx, rt60_0 in enumerate(rt60_0s[1:]):  # Excluding anechoic case
    result_mapped = np.asarray(median_RT_eval[rt60_0_idx]) * m + n
    mse[rt60_0_idx] = np.mean(np.power(rt60_0 - result_mapped, 2))
    mean_error[rt60_0_idx] = np.mean(rt60_0 - result_mapped)
    std_error[rt60_0_idx] = np.std(rt60_0 - result_mapped)

plt.subplot(326)
plt.title('Eval results - MSE')
plt.plot(rt60_0s[1:], mse, '-o', markersize=2)
plt.errorbar(rt60_0s[1:], mean_error, yerr=std_error, fmt='-o', markersize=2)


# %% all plots

plt.figure()
plt.suptitle(dataset + " - " + subset)
plt.subplot(321)
plt.title('Computed RT60')
plt.plot(results[:,1:], '-o', markersize=2)
plt.plot(results[:,0], '--', markersize=2)

# Difference wrt dry
diff = results[:,1:] - results[:,0,np.newaxis]
plt.subplot(322)
plt.title('Computed RT60 - Difference')
plt.plot(diff)

plt.subplot(323)
plt.title('masked results, th='+str(th))
for r in range(R-1):
    L = len(masked_results[r])
    plt.plot(masked_results[r], '-o', markersize=2, label=rt60_0s[r+1], color=colors[r])
    plt.hlines(rt60_0s[r+1], 0, maxN-1, linestyle='--', color=colors[r])
plt.legend()

plt.subplot(324)
plt.title('Masked results - stats & regression')
plt.errorbar(rt60_0s[1:], mean_values, yerr=std_values)
plt.plot(rt60_0s[1:], rt60_0s[1:], linestyle='--')
plt.plot(rt60_0s[1:], np.asarray(mean_values)*m+n, '-o', markersize=2)

plt.subplot(325)
plt.title('Eval results')
for rt60_0_idx, rt60_0 in enumerate(rt60_0s[1:]):  # Excluding anechoic case
    N = len(median_RT_eval[rt60_0_idx])
    # plt.plot(np.asarray(median_RT_eval[rt60_0_idx]), color=colors[rt60_0_idx])
    plt.plot(np.asarray(median_RT_eval[rt60_0_idx])*m+n, '-o', markersize=2, color=colors[rt60_0_idx], label=str(rt60_0))
    plt.hlines(rt60_0, 0, N-1, linestyle='--', color=colors[rt60_0_idx])
plt.legend()

plt.subplot(326)
plt.title('Eval results - MSE')
plt.plot(rt60_0s[1:], mse, '-o', markersize=2)
plt.errorbar(rt60_0s[1:], mean_error,  yerr=std_error, fmt='-o', markersize=2)

# %% RESULTS AS IN THE PAPER

# Statistics of results
mean_values_eval = []
for r in range(R-1):
    mean_values_eval.append(np.mean(median_RT_eval[r]))

# Correlation
slope, incercept, r_value, p_value, std_err = scipy.stats.linregress(np.asarray(mean_values_eval)*m+n, np.asarray(rt60_0s)[1:])
print(slope, incercept, r_value, p_value, std_err)



# %%
# %%
# %%
#
# # ALL RESULTS TOGETHER!!!!!
# #
# save
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
mr1 = np.load('/Users/andres.perez/source/dereverberation/blind_rt60/blind_rt60_data/masked_results_drummer_1.npy')
mr2 = np.load('/Users/andres.perez/source/dereverberation/blind_rt60/blind_rt60_data/masked_results_drummer_2.npy')
mr3 = np.load('/Users/andres.perez/source/dereverberation/blind_rt60/blind_rt60_data/masked_results_drummer_3.npy')
np.load = np_load_old

mr = [[] for i in range(4)]
for r in range(4):
    for v in mr1[r]:
        mr[r].append(v)
    for v in mr2[r]:
        mr[r].append(v)
    for v in mr3[r]:
        mr[r].append(v)



mean_values = []
std_values = []
for r in range(R-1):
    mean_values.append(np.mean(mr[r]))
    std_values.append(np.std(mr[r]))

plt.subplot(324)
plt.figure()
plt.title('Masked results - stats & regression')
plt.errorbar(rt60_0s[1:], mean_values, yerr=std_values)
plt.plot(rt60_0s[1:], rt60_0s[1:], linestyle='--')

# m, n, r_value, p_value, std_err = scipy.stats.linregress(mean_values, rt60_0s[1:])

x = np.asarray(rt60_0s[1:])
y = np.asarray(mean_values)
p0 = 2, 1 # initial guess
popt, pcov = scipy.optimize.curve_fit(line, y, x, p0, sigma=std_values, absolute_sigma=True)
yfit = line(y, *popt)
m2, n2 = popt
plt.plot(rt60_0s[1:], np.asarray(mean_values)*m2+n2, '-o', markersize=2)



