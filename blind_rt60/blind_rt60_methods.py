import numpy as np
import scipy.signal, scipy.optimize


def continuously_descendent(array):
    return np.all(array[:-1] > array[1:])

def line_origin(x, m):
    return m * x

def line(x, m, n):
    return m * x + n

def rt60_bands(rt60_0, nBands, decay=0.1):
    # decay per octave
    return np.asarray([rt60_0-(decay*i) for i in range(nBands)])


# %%  Main method by Prego et. al.
def estimate_blind_rt60(r_tf, sr=8000, window_overlap=256, FDR_time_limit=0.5):


    # %%  Subband FDR detection

    K, L = r_tf.shape
    e_tf = np.power(np.abs(r_tf), 2)  # energy spectrogram
    Llim = int(np.ceil(FDR_time_limit / (window_overlap / sr)))  # Number of consecutive windows to span FDR_time_lim

    # PAPER METHOD: at least one DFT for subband
    regions = []
    region_lens = []
    min_num_regions_per_band = 1  # this is per band!
    min_Llim = 3
    for k in range(K):
        num_regions_per_band = 0
        cur_Llim = Llim
        while num_regions_per_band < min_num_regions_per_band and cur_Llim >= min_Llim:
            cur_Llim -= 1
            num_regions_per_band = 0  # per band!
            for l in range(0, L - cur_Llim):
                if continuously_descendent(e_tf[k, l:l + cur_Llim]):
                    regions.append((k, l))
                    region_lens.append(cur_Llim)
                    num_regions_per_band += 1

    num_regions = len(regions)
    if num_regions == 0:
        raise ValueError('No FDR regions found!!')

    # PLOT -----
    # regions_tf = np.empty((K, L))
    # regions_tf.fill(np.nan)
    # for i, (k, l) in enumerate(regions):
    #     ll = region_lens[i]
    #     regions_tf[k, l:l + ll] = ll

    # plt.figure()
    # plt.title('Free Decay Regions')
    # plt.xlabel('Time (frames)')
    # plt.ylabel('Frequency (bins)')
    # plt.pcolormesh(20*np.log10(np.abs(r_tf)), vmin = -60, vmax = -20, cmap = 'inferno')
    # plt.pcolormesh(regions_tf)
    # print('number of regions:', num_regions)

    # %% 3. Subband Feature Estimation
    rt60s = np.zeros(num_regions)
    sedf = [[] for n in range(num_regions)]

    for FDR_region_idx, FDR_region in enumerate(regions):
        k, l = FDR_region
        Llim = region_lens[FDR_region_idx]
        den = np.sum(e_tf[k, l:l + Llim])

        for ll in range(Llim):
            num = np.sum(e_tf[k, l + ll:l + Llim])
            if num == 0: # filter out the TF bins with no energy
                Llim -= 1
            else:
                sedf[FDR_region_idx].append(10 * np.log10(num / den))

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

    # Final value: median of all subband medians
    return np.median(np.asarray(median_RT_per_subband))




# %% Attempt to write Lundeby's method

def lundeby(sedf, plot=False):
    # TODO: check...
    Llim = len(sedf)
    # 1. average: not needed
    # 2. last 10% of the impulse response
    x_bg = int(np.floor(Llim*0.9) -1) # preliminar truncation point
    delta_t = 10 # whatever
    max_iter = 20
    i = 0
    diff_dB = 5
    while (np.abs(delta_t) > 0) and (i < max_iter) and (x_bg < Llim):
        bg_noise_level = sedf[x_bg]
        # 3. linear regression between 0dB and bg_noise_level+5dB
        x_last_idx = len(sedf[sedf>(bg_noise_level+diff_dB)])-1
        if x_last_idx <= 0:
            break
        y_last_idx = sedf[x_last_idx]
        x = np.arange(x_last_idx+1)
        y = sedf[x]
        m = scipy.optimize.curve_fit(line_origin, x, y)[0][0]
        #4. crosspoint at the intersection with bg_noise_level
        x_cross = int(np.floor(bg_noise_level/m))
        delta_t = x_bg - x_cross
        # print(i, delta_t, x_bg, m)
        # 5. advance
        # x_bg = int(np.floor(x_bg - delta_t/2))
        x_bg = x_bg - 1
        i += 1

    if x_bg == Llim: # no noise floor
        x_bg = Llim - 1

    if plot:
        plt.figure()
        plt.plot(sedf)
        plt.grid()
        plt.vlines(x_bg, sedf[-1], 0)

    return sedf[:x_bg+1]



def generate_late_reverberation_ir_from_rt60(start_time, rt60, length, fs):
    """
    Returns a direct path (delta) at 0 delay plus a late reverberation
    in form of gaussian noise, with exponential decay.

    :param start_time: Time offset until the late reverb starts.
    :param rt60: rt60 time, in seconds
    :param length: length in seconds
    :param fs: sample rate
    :return: ir

    Example
    ________________
    start_time = 0.1
    rt60 = 1.5
    length = 2
    fs = 16000
    ir = generate_late_reverberation_ir_from_rt60(start_time, rt60, length, fs)

    plt.plot(np.arange(length*fs)/fs,ir)
    plt.grid()
    plt.show()

    plt.plot(np.arange(length*fs)/fs,10*np.log10(np.abs(ir)))
    plt.vlines(rt60,-100,0)
    plt.grid()
    plt.show()
    """

    num_samples = int(np.ceil(length * fs))
    start_sample = int(np.ceil(start_time * fs))
    n = np.arange(num_samples)
    num_samples_late_reverb = num_samples - start_sample

    ir = np.zeros(num_samples)
    ir[0] = 1.
    mean = 0.
    std = 1./3
    a = 3 * np.log(10) / rt60
    env = np.exp(-a*n[start_sample:]/fs) ** 2
    ir[start_sample:] = env * np.random.normal(mean, std, num_samples_late_reverb)

    return ir