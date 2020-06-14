"""
perform blind rt60 estimation on an audio track, based on the algorithm:

M. Prego, Thiago de, Amaro A. de Lima, Rafael Zambrano-Lopez, and Sergio L. Netto.
“Blind Estimators for Reverberation Time and Direct-to-Reverberant Energy Ratio Using Subband Speech Decomposition.”
In 2015 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), 1–5.
New Paltz, NY, USA: IEEE, 2015. https://doi.org/10.1109/WASPAA.2015.7336954.

which scored first on the ACE challenge 2015.

at the bottom, the linear mapping numbers are provided for further experiments


SAME AUDIO FILE, MULTIPLE ITERATIONS WITH DIFFERENT RT60s
"""

import numpy as np
import librosa
import scipy.signal
import matplotlib.pyplot as plt
from methods import plot_magnitude_spectrogram
from blind_rt60.datasets import get_audio_files

pp = "/Users/andres.perez/source/masp"
import sys
sys.path.append(pp)
import masp
from masp import shoebox_room_sim as srs
import os
import scipy.stats

# %% 0. Methods




def rt60_bands(rt60_0, nBands, decay=0.1):
    # decay per octave
    return np.asarray([rt60_0-(decay*i) for i in range(nBands)])





# %% 0. Prepare data

# get audio files
dataset = 'ENST'
subset = 'drummer_2'
main_path = '/Volumes/Dinge/datasets'
audio_files = get_audio_files(main_path, dataset, subset)

sr = 8000 # target sample rate
audio_file_length = 4.
audio_file_length_samples = int(audio_file_length * sr)

# TODO
audio_file_path = audio_files[2]
s_t = librosa.core.load(audio_file_path, sr=sr, mono=True)[0][:audio_file_length_samples]

rt60_decay = 0.05
nBands = 1
band_centerfreqs = np.empty(nBands)
band_centerfreqs[0] = 1000
for nb in range(1, nBands):
    band_centerfreqs[nb] = 2 * band_centerfreqs[nb - 1]

sh_order = 0
dimM = (sh_order+1)**2


# %% 0. Reverberant signal

I = 21

median_RT = np.zeros(I)
median_rt60s = np.zeros(I)
mean_rt60s = np.zeros(I)
rt60_1kHz = np.zeros(I) # rt60 at 1kHz

for iter in range(I):

    print('--------------------------------------------')
    print('ITER:', iter)
    rt60_0 = np.random.rand()*0.5 + 0.3

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
    src_sph = np.array([azi, np.pi / 2 - incl, d_critical.mean()/2])
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

    r_t = scipy.signal.fftconvolve(s_t, irs[0])[:audio_file_length_samples]  # keep original length

    # %% 1. Time-frequency representation

    window_size = 1024
    window_overlap = window_size // 4
    nfft = window_size

    f, t, r_tf = scipy.signal.stft(r_t, sr, nperseg=window_size, noverlap=window_overlap, nfft=nfft)
    K, L = r_tf.shape

    _, _, s_tf = scipy.signal.stft(s_t, sr, nperseg=window_size, noverlap=window_overlap, nfft=nfft)

    # plt.figure()
    # plot_magnitude_spectrogram(s_tf, title='Anechoic STFT' )
    # plot_magnitude_spectrogram(r_tf, title='Reverberant STFT' )


    # %% 2. Subband FDR detection

    e_tf = np.power(np.abs(r_tf), 2) # energy spectrogram

    FDR_time_lim = 0.5 # 500 ms
    Llim = int(np.ceil(FDR_time_lim / (window_overlap / sr))) # Number of consecutive windows to span FDR_time_lim

    # PAPER METHOD: at least one DFT for subband
    regions = []
    region_lens = []
    min_num_regions_per_band = 1 # this is per band!
    min_Llim = 3
    for k in range(K):
        num_regions_per_band = 0
        cur_Llim = Llim
        while num_regions_per_band < min_num_regions_per_band and cur_Llim >= min_Llim:
            cur_Llim -= 1
            num_regions_per_band = 0 # per band!
            for l in range(0, L - cur_Llim):
                if continuously_descendent(e_tf[k,l:l+cur_Llim]):
                    regions.append((k,l))
                    region_lens.append(cur_Llim)
                    num_regions_per_band += 1

    num_regions = len(regions)
    regions_tf = np.empty((K, L))
    regions_tf.fill(np.nan)

    for i, (k, l) in enumerate(regions):
        ll = region_lens[i]
        regions_tf[k,l:l+ll] = ll
        # regions_tf[k, l] = ll

    # PLOT -----
    # plt.figure()
    # plt.title('Free Decay Regions')
    # plt.xlabel('Time (frames)')
    # plt.ylabel('Frequency (bins)')
    # plt.pcolormesh(20*np.log10(np.abs(r_tf)), vmin = -60, vmax = -20, cmap = 'inferno')
    # plt.pcolormesh(regions_tf)
    # print('number of regions:', num_regions)

    # OLD METHOD: num min of regions in all spectrum
    # # find continuously decreasing regions, decrease Llim if not found
    # num_regions = 0
    # min_num_regions = 5
    # while num_regions < min_num_regions:
    #     print('Llim:', Llim)
    #     Llim -= 1
    #     regions = []
    #     for k in range(K):
    #         for l in range(0, L-Llim):
    #             if continuously_descendent(e_tf[k,l:l+Llim]):
    #                 regions.append((k,l))
    #     num_regions = len(regions)
    #
    # regions_tf = np.empty((K, L))
    # regions_tf.fill(np.nan)
    #
    # for k, l in regions:
    #     # regions_tf[k,l:l+Llim] = 100
    #     regions_tf[k, l] = 100
    #
    # # plt.figure()
    # # plt.pcolormesh(regions_tf)
    # print('number of regions:', num_regions)

    # %% 3. Subband Feature Estimation

    rt60s = np.zeros(num_regions)
    # sedf = np.zeros((num_regions, Llim))
    sedf = [ [] for n in range(num_regions)]

    for FDR_region_idx, FDR_region in enumerate(regions):
        k, l = FDR_region
        Llim = region_lens[FDR_region_idx]
        den = np.sum(e_tf[k, l:l+Llim])

        for ll in range(Llim):
            num = np.sum(e_tf[k, l+ll:l+Llim])
            sedf[FDR_region_idx].append(10 * np.log10( num / den ))

        # sedf_lundeby = lundeby(sedf, plot=True)
        # sedf_lundeby = sedf[FDR_region_idx]
        # Fit linear decay to the lundeby version
        # m = scipy.optimize.curve_fit(line_origin, np.arange(sedf_lundeby.size), sedf_lundeby)[0][0]

        m = scipy.optimize.curve_fit(line_origin, np.arange(Llim), sedf[FDR_region_idx])[0][0]
        y2 = -60
        x2 = y2 / m  # in number of windows
        rt60s[FDR_region_idx] = x2 * window_overlap / sr  # in seconds

    # plt.figure()
    # plt.plot(sedf[3], label='True SEDF')
    # plt.grid()
    # plt.plot(np.arange(9), np.arange(9) * m, linestyle='--', label='Linear fit')
    # plt.title('Free Decay Region')
    # plt.xlabel('Time frames')
    # plt.ylabel('Subband Energy Decay Function')
    # plt.legend()
    # decay curves

    # stats of rt60s
    # plt.hist(rt60s, bins = np.arange(0,1,0.05))
    # plt.grid()
    # plt.xlabel('RT60')
    # plt.title('FDR Histogram of estimated RT60s')

    median_rt60s[iter] = np.median(rt60s)
    mean_rt60s[iter] = np.mean(rt60s)

    # %% 4. Statistical analysis of subbands RTs

    # Compute the RT(k) as the median of all RT estimates of subband k

    RT_per_subband = [ [] for k in range(K)] # init to empty list of lists„

    for FDR_region_idx, FDR_region in enumerate(regions): # group decay estimates by frequency bin
        k, _ = FDR_region
        RT_per_subband[k].append(rt60s[FDR_region_idx])

    # take the median
    median_RT_per_subband = []
    for k in range(K):
        rts = RT_per_subband[k]
        if len(rts) > 0: # not empty
            median_RT_per_subband.append(np.median(rts))

    # Final value: median of all medians
    median_RT[iter] = np.median(np.asarray(median_RT_per_subband))
    # rt60 at 1kHz
    rt60_1kHz[iter] = rt60[0]
    # difference
    # diff = median_RT - rt60_1kHz

# plt.figure()
# plt.ylim(0, 1.5)
# plt.plot(np.arange(K)/K*sr/2, median_RT_per_subband)
# plt.plot(band_centerfreqs/band_centerfreqs[-1]*sr/2, rt60, '.-')
# plt.hlines(median_RT, 0, sr/2)
# plt.xlabel('frequency')
# plt.ylabel('estimated RT60')
# plt.grid()



# %% Find correlation

iii = np.argsort(rt60_1kHz)

plt.figure()
plt.title('True and estimated RT60')
plt.plot(np.arange(I), rt60_1kHz[iii], '-o', label='true')
plt.plot(np.arange(I), median_RT[iii], '-o', label='median_RT')
plt.plot(np.arange(I), median_rt60s[iii], '-o', label='median_rt60s')
plt.plot(np.arange(I), mean_rt60s[iii], '-o', label='mean_rt60s')
plt.grid()
plt.legend()
plt.xlabel('Iteration number')
plt.ylabel('RT60')


print("m \t\t\t\t\t n \t\t\t\t\t r_value \t\t\t p_value \t\t\t std_err")
print('--------------------------------------------------------------'
      '--------------------------------------------------------------')
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(median_RT, rt60_1kHz)
print(slope, intercept, r_value, p_value, std_err)
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(median_rt60s, rt60_1kHz)
print(slope, intercept, r_value, p_value, std_err)
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(mean_rt60s, rt60_1kHz)
print(slope, intercept, r_value, p_value, std_err)

## I = 101, 10 seconds, 0.2..0.7, drums
# m 					 n 					 r_value 			 p_value 			 std_err
# ----------------------------------------------------------------------------------------------------------------------------
# 6.618403381439503 -1.7247530528506398 0.9613277036972795 3.0014644971266755e-57 0.19056328886056154
# 6.276508977283504 -1.4951470547984325 0.963644372221467 1.4917157216356556e-58 0.17490451091121864
# 7.057618730501374 -1.8951377003042544 0.9669863552433601 1.3671021160678983e-60 0.18692511662127417


# %% Compute data

# use medianRT values
m, n = scipy.stats.linregress(median_RT, rt60_1kHz) [:2]

plt.figure()
plt.title('True and estimated RT60, after mapping')
plt.grid()
plt.plot(np.arange(I), rt60_1kHz[iii], '-o', label='true')
plt.plot(np.arange(I), median_RT[iii]*m + n, '-o', label='median_RT')
plt.legend()
plt.xlabel('Iteration number')
plt.ylabel('RT60')

mse = np.mean(np.power(rt60_1kHz[iii]-(median_RT[iii]*m + n), 2))

# stats
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(median_RT*m + n, rt60_1kHz)

print("AFTER MAPPING")
print('----------------------------------------')
print("std_err \t\t\t\t\t MSE \t\t\t\t\t ")
print(std_err, mse)