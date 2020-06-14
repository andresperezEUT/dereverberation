"""
esperiment 3

ANALYSE ONE AUDIO, EXTRACT RT60, REREVERBERATE IT (MONO), CREATE AUDIOS TO LISTEN
"""
import warnings
import csv

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
import random
import soundfile as sf

plt.style.use('seaborn-whitegrid')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # default colormap


# %% 0. Prepare data

# get audio files
num_files = -1
dataset = 'DSD100'
subset = ''
main_path = '/Volumes/Dinge/datasets'
audio_files = get_audio_files(main_path, dataset, subset)[:num_files]

sr = 48000 # target sample rate
downsampled_sr = 8000
dw_factor = sr/downsampled_sr


rt60_decay = 0.05
nBands = 1
band_centerfreqs = np.empty(nBands)
band_centerfreqs[0] = 1000
for nb in range(1, nBands):
    band_centerfreqs[nb] = 2 * band_centerfreqs[nb - 1]

sh_order = 1
dimM = (sh_order + 1) ** 2

N = len(audio_files)

I = 2 # num repetitions

for iter in range(I):

    print('--------------------------------------------')
    print('ITER: ', iter)

    # %% ROOM

    rt60_0 = np.random.rand() * 0.6 + 0.4
    print('RT60: ', rt60_0)

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



    # %% ANALYSIS

    audio_file_length = 8.
    audio_file_length_samples = int(audio_file_length * sr)
    audio_file_offset = 2.
    audio_file_offset_samples = int(audio_file_offset * sr)
    af_start = audio_file_offset_samples
    af_end = audio_file_offset_samples + audio_file_length_samples

    # for dsd100
    valid_audio_file_idx = np.array([ 0,  1,  2,  4,  5,  6,  8,  9, 11, 12, 15, 16, 18, 19, 20, 22, 23,
           24, 29, 31, 32, 33, 34, 35, 37, 38, 39, 41, 42, 43, 44, 46, 47, 48,
           49, 51, 52, 53, 54, 56, 57, 58, 59, 60, 63, 65, 67, 68, 69, 72, 74,
           77, 79, 80, 81, 83, 84, 85, 86, 87, 88, 89, 91, 92, 94, 95, 96, 98])


    window_size = 1024
    window_overlap = window_size // 4
    nfft = window_size

    FDR_time_limit = 0.5
    # m, n = (3.4234199209082963, -0.45679586527031363)
    m, n = (6.123920485012435, -1.2566895887421592)

    finish = False
    while not finish:

        audio_file_idx = random.choice(valid_audio_file_idx)
        audio_file_path = audio_files[audio_file_idx]
        s_t = librosa.core.load(audio_file_path, sr=sr, mono=True)[0][af_start:af_end]
        print('FILE:', audio_file_path)

        r_t = scipy.signal.fftconvolve(s_t, h_t)[:audio_file_length_samples]  # keep original length

        # Analysis performed on the downsampled version
        r_t_dw = [ v for i, v in enumerate(r_t) if i%dw_factor==0]
        f, t, r_tf = scipy.signal.stft(r_t_dw, downsampled_sr, nperseg=window_size, noverlap=window_overlap, nfft=nfft)

        try:
            est_rt60_0 = estimate_blind_rt60(r_tf, downsampled_sr, window_overlap, FDR_time_limit) * m + n
            if est_rt60_0 > 0:
                finish = True
        except ValueError:
            warnings.warn('no FDR')



    # %% SYNTHESIS

    print('estimated rt60: ', est_rt60_0)
    print('RESYNTHESIS:')

    room = np.array([10.2, 7.1, 3.2])
    rt60 = rt60_bands(est_rt60_0, nBands, rt60_decay)

    abs_wall = srs.find_abs_coeffs_from_rt(room, rt60)[0]

    # Critical distance for the room
    _, d_critical, _ = srs.room_stats(room, abs_wall, verbose=False)


    maxlim = rt60_0  # the original one
    limits = np.ones(nBands) * maxlim  # hardcoded!

    abs_echograms = srs.compute_echograms_sh(room, src, rec, abs_wall, limits, rec_orders)
    est_irs = srs.render_rirs_sh(abs_echograms, band_centerfreqs, sr).squeeze().T
    if est_irs.ndim == 1:
        est_irs = est_irs[np.newaxis, :]
    # Normalize as SN3D
    est_irs *= np.sqrt(4 * np.pi)
    est_h_t = est_irs[0]
    est_r_t = scipy.signal.fftconvolve(s_t, est_h_t)[:audio_file_length_samples]  # keep original length

    # Synthesize ambisonic audios
    r_t_ambi = np.empty((audio_file_length_samples,dimM))
    est_r_t_ambi = np.empty((audio_file_length_samples,dimM))
    for m in range(dimM):
        r_t_ambi[:,m] = scipy.signal.fftconvolve(s_t, irs[m])[:audio_file_length_samples]  # keep original length
        est_r_t_ambi[:,m] = scipy.signal.fftconvolve(s_t, est_irs[m])[:audio_file_length_samples]  # keep original length

    # Create folder
    new_folder_path = os.path.join('/Users/andres.perez/source/dereverberation/blind_rt60/blind_rt60_audio', str(iter))
    if os.path.exists(new_folder_path):
        os.rmdir(new_folder_path)
    os.mkdir(new_folder_path)

    # Create metadata file
    new_file_path = os.path.join(new_folder_path, 'metadata.csv')
    with open(new_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["azi", (azi - np.pi)*360/(2*np.pi)])
        writer.writerow(["ele", (np.pi/2 - incl)*360/(2*np.pi)])
        writer.writerow(["rt60", rt60_0])
        writer.writerow(["est_rt60", est_rt60_0])
        writer.writerow(["filename", audio_file_path])
        writer.writerow(["audio_file_length", audio_file_length])
        writer.writerow(["audio_file_offset", audio_file_offset])
        writer.writerow(["sr", sr])
        writer.writerow(["sr_dodownsampled_srwnsample", downsampled_sr])

    # Write audio
    sf.write(os.path.join(new_folder_path, 'dry.wav'), s_t, sr)
    sf.write(os.path.join(new_folder_path, 'true.wav'), r_t_ambi, sr)
    sf.write(os.path.join(new_folder_path, 'est.wav'), est_r_t_ambi, sr)

    print('azi, ele')
    print((azi - np.pi)*360/(2*np.pi)) # compensate for left-right change
    print((np.pi/2 - incl)*360/(2*np.pi))


# %% LATE REVERB APPROX

# start_time = 0.1
# length = 1
# late_h_t = generate_late_reverberation_ir_from_rt60(start_time, est_rt60_0, length, sr)
# est_late_r_t = scipy.signal.fftconvolve(s_t, late_h_t)[:audio_file_length_samples]  # keep original length
# sf.write('/Users/andres.perez/source/dereverberation/blind_rt60/blind_rt60_audio/late_est.wav', est_late_r_t, sr)
