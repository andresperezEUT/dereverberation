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
# from ctf_methods import sid_stft2 # TODO

from ctf.ctf_methods import sid_stft2, rt60_bands, compute_t60
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

sh_order = 1
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

audio_file_length = 5.  ## seconds
audio_file_length_samples = int(audio_file_length * fs)

## MAR-------
# D = int(np.floor(ir_start_time * fs / window_overlap)) # ir start frame
tau = int(1/hop)
print('tau=',tau)
if tau < 1:
    raise Warning('D should be at least 1!!!')

# noise_power = 1e-5

# get audio files
audio_files = []
data_folder_path = '/Volumes/Dinge/subset_DSD100/Sources'
for root, dir, files in os.walk(data_folder_path):
    for f in files:
        extension = os.path.splitext(f)[1]
        if 'wav' in extension:
            audio_files.append(os.path.join(root, f))


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

I = 3  # number of repetitions

rt_methods = ['edt', 't10', 't20', 't30']
R = len(rt_methods)

rt60_1kHz = np.zeros(I) # true rt60 at 1kHz
rt60_ir_true = np.zeros((I,R))
rt60_ir_est_true = np.zeros((I,R))
rt60_ir_est_derv = np.zeros((I,R))

L = fs # todo: adjust here filter length
ir_true = np.zeros((I, L))
ir_est_true = np.zeros((I, L))
ir_est_derv = np.zeros((I, L))



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

    af = audio_files[1]
    # af = '/Volumes/Dinge/audio/410298__inspectorj__voice-request-26b-algeria-will-rise-again-serious.wav'
    #
    # # Open audio files and encode into ambisonics

    mono_s_t = librosa.core.load(af, sr=fs, mono=True)[0][:audio_file_length_samples]
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

    import matlab.engine
    eng = matlab.engine.start_matlab()
    matlab_path = '/Users/andres.perez/source/dereverberation/ctf/akis_ctfconv'
    eng.addpath(matlab_path)


    ir_true[iter] = irs[0]  # true impulse response
    filtersize = ir_true[iter].size # true, groundtruth
    winsize = 8 * filtersize
    hopsize = winsize / 16

    # IR ESTIMATION FROM TRUE ANECHOIC SIGNAL
    ir_est_true[iter] = sid_stft2(s_t, y_t[0], winsize, hopsize, filtersize)

    # IR ESTIMATION FROM ESTIMATED ANECHOIC SIGNAL
    ir_est_derv[iter] = sid_stft2(est_s_t[0], y_t[0], winsize, hopsize, filtersize)

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
    est_true = compute_t60(ir_est_true[iter], fs, band_centerfreqs, plot=plot, title='Estimated IR, true signal')
    est_derv = compute_t60(ir_est_derv[iter], fs, band_centerfreqs, plot=plot, title='Estimated IR, derv signal')

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
    rt60_ir_est_true[iter] = est_true.squeeze()
    rt60_ir_est_derv[iter] = est_derv.squeeze()




# %% results

iii = np.argsort(rt60_1kHz)

fig = plt.figure(figsize=plt.figaspect(1/3))
plt.subplot(131)
plt.title('Estimated RT60, True IR')
plt.plot(np.arange(I), rt60_1kHz[iii], '-o', label='true')
for rt_method_idx, rt_method in enumerate(rt_methods):
    plt.plot(np.arange(I), rt60_ir_true[iii, rt_method_idx], '-o', label=rt_method)

plt.subplot(132)
plt.title('Estimated rt60, true anechoic signal')
plt.plot(np.arange(I), rt60_1kHz[iii], '-o', label='true')
for rt_method_idx, rt_method in enumerate(rt_methods):
    plt.plot(np.arange(I), rt60_ir_est_true[iii, rt_method_idx], '-o', label=rt_method)
plt.legend()
plt.xlabel('Iteration number')
plt.ylabel('RT60')

plt.subplot(133)
plt.title('Estimated rt60, dereverberated signal')
plt.plot(np.arange(I), rt60_1kHz[iii], '-o', label='true')
for rt_method_idx, rt_method in enumerate(rt_methods):
    plt.plot(np.arange(I), rt60_ir_est_derv[iii, rt_method_idx], '-o', label=rt_method)



# %% Compute data

print("m \t\t\t\t\t n \t\t\t\t\t r_value \t\t\t p_value \t\t\t std_err")
print('--------------------------------------------------------------'
      '--------------------------------------------------------------')
for rt_method_idx in range(len(rt_methods)):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(rt60_ir_est_derv[:,rt_method_idx], rt60_1kHz)
    print(slope, intercept, r_value, p_value, std_err)

# m 					 n 					 r_value 			 p_value 			 std_err
# ----------------------------------------------------------------------------------------------------------------------------
# 0.8081623460458246 0.1325598312669864 0.9142455212288917 7.339057647514432e-21 0.051164054871867926
# 0.7025215578059423 -0.007794992567297598 0.9402900028321771 1.4000419656274352e-24 0.03632926782658806
# 0.630424354852478 -0.24819500408764805 0.8961299514523163 6.48592594809326e-19 0.04460085314221639
# -1.2224161039695636 2.5201305818233943 -0.28787742423158513 0.04051327283935966 0.5809356876377273


rt_method_idx = 0
m, n = scipy.stats.linregress(rt60_ir_est_derv[:,rt_method_idx], rt60_1kHz) [:2]

plt.figure()
plt.title('True and estimated RT60, after mapping')
plt.grid()
plt.plot(np.arange(I), rt60_1kHz[iii], '-o', label='true')
plt.plot(np.arange(I), rt60_ir_est_derv[iii, rt_method_idx]*m + n, '-o', label='t10')
plt.legend()
plt.xlabel('Iteration number')
plt.ylabel('RT60')


# stats
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(rt60_ir_est_derv[:, rt_method_idx]*m + n, rt60_1kHz)
print(slope, intercept, r_value, p_value, std_err)