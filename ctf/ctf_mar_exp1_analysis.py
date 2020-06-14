"""
esperiment 1 - DATA ANALYSIS

DSD set, seconds 1-5, 8 kHz, L=5, imax=10, w=128 0.4-1 s rt
save data in /ctf_data
"""



import matplotlib.pyplot as plt
import scipy.signal
import numpy as np
cmap = plt.get_cmap("tab20c")
plt.style.use('seaborn-whitegrid')

from blind_rt60.datasets import get_audio_files


# %% PARAMETERS

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
ir_length_seconds = rt60_0s[-1]
ir_length_samples = int(ir_length_seconds*sr)
rt_methods = ['edt', 't10', 't20', 't30']
Ls = [5]  # number of frames for the IIR filter
R = len(rt60_0s)
N = len(audio_files)




colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def index_of_first_element(array, cond):
    # first element bigger or equal than cond, or nan if none
    index = np.nonzero(array >= cond)[0][-1] + 1
    if index == array.size:
        return np.nan
    else:
        return index

def get_sch(ir, tau, window_overlap):
    band = 1000
    flim = [band / np.sqrt(2), band * np.sqrt(2)]
    sos = scipy.signal.butter(3, flim, btype='bandpass', output='sos', fs=sr)
    ir = ir[1:]
    filtered_ir = scipy.signal.sosfilt(sos, ir)
    a_t = np.abs(scipy.signal.hilbert(filtered_ir))
    sch = np.cumsum(a_t[::-1] ** 2)[::-1]
    sch_db = 10.0 * np.log10(sch / np.max(sch))
    sch_db = sch_db[:int(0.8*sch_db.size)]
    return sch_db, sch, a_t


def fit_rt60(data, sr):
    rt_methods = ['edt', 't10', 't20', 't30']
    rt60 = np.empty(len(rt_methods))
    ms = np.empty(len(rt_methods))
    ns = np.empty(len(rt_methods))
    for rt_method_idx, rt_method in enumerate(rt_methods):
        if rt_method == 'edt':
            m = -10 / index_of_first_element(data, -10)  # dB / sample
            n = 0  # offset
        elif rt_method == 't10':
            a = index_of_first_element(data, -5)
            b = index_of_first_element(data, -15)
            m = (15 - 5) / (a - b)
            n = -5 - m * a  # - n = mx + y
        elif rt_method == 't20':
            a = index_of_first_element(data, -5)
            b = index_of_first_element(data, -25)
            m = (25 - 5) / (a - b)
            n = -5 - m * a  # - n = mx + y
        elif rt_method == 't30':
            a = index_of_first_element(data, -5)
            b = index_of_first_element(data, -35)
            m = (35 - 5) / (a - b)
            n = -5 - m * a  # - n = mx + y
        rt60_value = (-60 - n) / m / sr
        rt60[rt_method_idx] = rt60_value
        ms[rt_method_idx] = m
        ns[rt_method_idx] = n
    return rt60, ms, ns


def line(x, m, n):
    return m * x + n
# %% ANALYSIS

sh_order = 0
dimM = (sh_order+1)**2
tau = 1

# SHAPE: ((R, N, len(Ls), dimM, ir_length_samples))
irs_estimated = np.load('/Users/andres.perez/source/dereverberation/ctf/ctf_data/irs_estimated_'+str(sh_order)+'_1.npy')
# irs_estimated = irs_estimated[::3]

rt60_0s = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# rt60_0s = rt60_0s[::3]
R = len(rt60_0s)
l_idx = 0

audio_files = audio_files[np.array([0,10,17,18,23,30,40,42,49,53,54,56,57])]
N = len(audio_files)




# # %% ANALYSIS - CHECK ONE
# af_idx = 4
# ch = 0
#
# l_idx = 0
# nb = 0
# band = band_centerfreqs[nb]
# flim = [band / np.sqrt(2), band * np.sqrt(2)]
# sos = scipy.signal.butter(3, flim, btype='bandpass', output='sos', fs=sr)
# method_idx = 0
#
#
# L = irs_estimated.shape[-1]
# plt.figure()
# plt.suptitle('af_idx: '+ str(af_idx) + ', method_idx: '+str(method_idx))
# end_samples = [L, int(L/2), int(L/4), int(L/8)]
# for idx, end_sample in enumerate(end_samples):
#     plt.subplot(2,2,idx+1)
#     plt.title(str(end_sample))
#     for rt_idx, rt in enumerate(rt60_0s):
#         true_rt60 = rt60_0s[rt_idx]
#         ir = irs_estimated[rt_idx, af_idx, l_idx, ch, :end_sample]
#         sch_db = get_sch(ir, tau, window_overlap)[0]
#         rt60_value, m, n = fit_rt60(sch_db, sr)
#         print('-----------------------------')
#         print('true:      ', true_rt60)
#         print('estimated: ', rt60_value[method_idx])
#
#         x = np.arange(len(sch_db))
#         y = m[method_idx]*x + n[method_idx]
#         m_true = -60 / true_rt60 / sr
#         plt.plot(x, x * m_true, label='true ' + str(rt), linestyle='-.', )
#
#         plt.plot(sch_db, label='true '+str(rt), color=colors[rt_idx])
#         plt.plot(x, y, label=rt_methods[method_idx]+str(rt), linestyle='--', linewidth=1, color=colors[rt_idx])
#         plt.ylim(max(sch_db)-60,max(sch_db)+10)
# plt.legend()
#
#
# rt_idx = 0
# rt = rt60_0s[rt_idx]
# plt.figure()
# plt.suptitle('af_idx: '+ str(af_idx))
# for idx, end_sample in enumerate(end_samples):
#     plt.subplot(2,2,idx+1)
#     plt.title(str(end_sample))
#     true_rt60 = rt60_0s[rt_idx]
#     ir = irs_estimated[rt_idx, af_idx, l_idx, ch, :end_sample]
#     sch_db, sch, a_t= get_sch(ir, tau, window_overlap)
#     plt.plot(a_t, label=str(rt), linewidth=1, color=colors[0])
#     plt.plot(sch, label=str(rt), linestyle='--', linewidth=1, color=colors[1])
#
#
#
# # shape: (r, end_samples, method)
# estimated_rt60 = np.empty((R, 4, 4))
#
#
# end_samples = [L, int(L/2), int(L/4), int(L/8)]
# for idx, end_sample in enumerate(end_samples):
#     for rt_idx, rt in enumerate(rt60_0s):
#         print(idx, rt_idx)
#         true_rt60 = rt60_0s[rt_idx]
#         ir = irs_estimated[rt_idx, af_idx, l_idx, ch, :end_sample]
#         sch_db = get_sch(ir, tau, window_overlap)[0]
#         fit_values, ms, ns = fit_rt60(sch_db,sr)
#         print(fit_values)
#         estimated_rt60[rt_idx, idx] = fit_values
#
# rt_methods = ['edt', 't10', 't20', 't30']
# plt.figure()
# plt.suptitle('af_idx: '+ str(af_idx))
# for idx, end_sample in enumerate(end_samples):
#     plt.subplot(2,2,idx+1)
#     plt.title(str(end_sample))
#     plt.plot(rt60_0s, rt60_0s, linestyle='--', linewidth=1, color=colors[0])
#     for method_idx, method in enumerate(rt_methods):
#         plt.plot(rt60_0s, estimated_rt60[:,idx,method_idx], label=method, linewidth=1, color=colors[method_idx+1])
# plt.legend()


# %% ANALYSIS
# mean for all n


#
# # shape: r, n, end_samples, method)
# estimated_rt60 = np.empty((R, N, 4, 4, dimM))
# perr = np.empty((4, 4, dimM))
# param = np.empty((4, 4, 2, dimM))
# perr.fill(np.inf)
# param.fill(np.inf)
#
# for ch in range(dimM):
#     print('channel: '+str(ch))
#
#     L = irs_estimated.shape[-1]
#     end_samples = [L, int(L/2), int(L/4), int(L/8)]
#
#     for af_idx in range(N):
#         for idx, end_sample in enumerate(end_samples):
#             for rt_idx, rt in enumerate(rt60_0s):
#                 true_rt60 = rt60_0s[rt_idx]
#                 ir = irs_estimated[rt_idx, af_idx, l_idx, ch, :end_sample]
#                 sch_db = get_sch(ir, tau, window_overlap)[0]
#                 fit_values, ms, ns = fit_rt60(sch_db,sr)
#                 estimated_rt60[rt_idx, af_idx, idx, :, ch] = fit_values
#
#
#     meanvalue = np.nanmean(estimated_rt60, axis=1)
#     stdvalue = np.nanstd(estimated_rt60, axis=1)
#
#     rt_methods = ['edt', 't10', 't20', 't30']
#     plt.figure()
#     plt.suptitle('mean of all samples, ch: '+str(ch))
#     for idx, end_sample in enumerate(end_samples):
#         plt.subplot(2,2,idx+1)
#         plt.title(str(end_sample))
#         plt.plot(rt60_0s, rt60_0s, linestyle='--', linewidth=1, color=colors[0])
#         for method_idx, method in enumerate(rt_methods):
#             plt.errorbar(rt60_0s, meanvalue[:,idx,method_idx,ch], yerr=stdvalue[:,idx,method_idx,ch], label=method, linewidth=1, color=colors[method_idx+1])
#     plt.legend()
#
#
#     for idx, end_sample in enumerate(end_samples):
#         for method_idx, method in enumerate(rt_methods):
#             x = rt60_0s
#             y = meanvalue[:, idx, method_idx, ch]
#             sigma = stdvalue[:, idx, method_idx, ch]
#             if not np.any(np.isnan(y)) and not np.any(np.isnan(sigma)):
#                 p0 = [1,0]
#                 popt, pcov = scipy.optimize.curve_fit(line, y, x, p0, sigma=sigma, absolute_sigma=True)
#                 param[idx, method_idx, :,ch] = popt
#                 var = np.sum(np.diag(pcov))
#                 std = np.sqrt(var) # joint standard deviation is sqrt of sum of variances https://socratic.org/statistics/random-variables/addition-rules-for-variances
#                 perr[idx, method_idx, ch] = std
#
#     # best fitting is the one minimizing joint std dev
#     best_fit = np.unravel_index(perr[:,:,ch].argmin(), perr[:,:,ch].shape)
#     best_end_sample, best_rt_method = best_fit
#     print('best fit:', best_fit)
#     print('best_end_sample fit:', end_samples[best_end_sample])
#     print('best_rt_method fit:', rt_methods[best_rt_method])
#     print('std dev: ', perr[best_end_sample, best_rt_method, ch])
#     print('param: ', param[best_end_sample, best_rt_method, :, ch])
#
# plt.figure()
# for ch in range(dimM):
#     plt.subplot(np.sqrt(dimM), np.sqrt(dimM), ch+1)
#     plt.title(str(ch))
#     plt.imshow(perr[:,:,ch], cmap='magma')
#     plt.colorbar()
#     plt.xlabel('method')
#     plt.ylabel('length')
#


#%% ######

estimated_edt = np.empty((R, N))

ch = 0
l_idx = 0
idx = 0
end_sample = end_samples[idx]
method_idx = 0
for af_idx in range(N):
    for rt_idx, rt in enumerate(rt60_0s):
        true_rt60 = rt60_0s[rt_idx]
        ir = irs_estimated[rt_idx, af_idx, l_idx, ch, :end_sample]
        sch_db = get_sch(ir, tau, window_overlap)[0]
        fit_values, ms, ns = fit_rt60(sch_db,sr)
        estimated_edt[rt_idx, af_idx] = fit_values[method_idx]


mean = np.nanmean(estimated_edt,axis=1)
std = np.nanstd(estimated_edt,axis=1)
plt.figure()
for rt_idx, rt in enumerate(rt60_0s):
    plt.subplot(2,4,rt_idx+1)
    plt.plot(np.arange(N), estimated_edt[rt_idx])
    plt.hlines(rt, 0, N-1, linestyle='--', linewidth=0.5)
    plt.hlines(mean[rt_idx], 0, N-1, linestyle='--', linewidth=1, color=colors[0])
    plt.hlines(mean[rt_idx]+std[rt_idx], 0, N-1, linestyle='--', linewidth=0.75, color=colors[0])
    plt.hlines(mean[rt_idx]-std[rt_idx], 0, N-1, linestyle='--', linewidth=0.75, color=colors[0])

    plt.ylim(0,1.5)
#%% ###### RESULTS

# shorder 0

# START 0, END -1
# best fit: (2, 3)
# best_end_sample fit: 2000
# best_rt_method fit: t30
# std dev:  1.6885445825370986
# param:  [ 8.71522067 -2.59704071]

# START 0, END 0.9
# best fit: (2, 3)
# best_end_sample fit: 2000
# best_rt_method fit: t30
# std dev:  1.6445916785669303
# param:  [ 9.61317797 -2.81103484]

# # # # # # # # # # # # # # # # # #
# START0 tau * window_overlap, end -1
# best fit: (2, 2)
# best_end_sample fit: 2000
# best_rt_method fit: t20
# std dev:  1.560211363308321
# param:  [ 6.90130951 -2.49325912]
# # # # # # # # # # # # # # # # # #

# START0 tau * window_overlap, end 0.9
# best fit: (2, 2)
# best_end_sample fit: 2000
# best_rt_method fit: t20
# std dev:  1.5910275922926913
# param:  [ 7.2053627  -2.54340856]



######

# sh 1, m 0
# best fit: (2, 0)
# best_end_sample fit: 2000
# best_rt_method fit: edt
# std dev:  0.698155907371107
# param:  [ 1.51714422 -0.53488281]