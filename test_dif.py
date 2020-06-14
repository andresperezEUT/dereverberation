import sys
pp = "/Users/andres.perez/source/parametric_spatial_audio_processing"
sys.path.append(pp)
import parametric_spatial_audio_processing as psa
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import os
import librosa
import librosa.display
import pyaudio
import time

pp = "/Users/andres.perez/source/masp"
sys.path.append(pp)
import masp
from methods import *


audio_files = []
data_folder_path = '/Volumes/Dinge/DSD100subset/Sources'
for root, dir, files in os.walk(data_folder_path):
    for f in files:
        extension = os.path.splitext(f)[1]
        if 'wav' in extension:
            audio_files.append(os.path.join(root, f))




L=1 ## ambisonics order
# TODO: real uniform along the sphere
azi = np.random.rand()*2*np.pi
incl = np.random.rand()*np.pi
print('AZI - ELE', azi, np.pi/2 - incl)
dirs = np.asarray([[azi, incl]])
basisType = 'real'
y = masp.get_sh(L, dirs, basisType) * np.sqrt(4*np.pi) * [1, 1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)] ## ACN, SN3D

## GET FOUR RANDOM IRS, ONE FOR EACH CHANNEL, AND APPLY GAINS TO THE DIRECT PATH
## the late reverb should keep the same amplitude (energy) regardless of the direction

M = np.power(L+1, 2)

start_time = 0.01
length = 1
fs = 48000
irs = np.zeros((M, length*fs))

# rt60_vector = np.arange(0.1, 1.6, 0.1)
rt60_vector = [0.1]
for af in [audio_files[1]]: # todo
    lsdi_vector = []
    for rt60 in rt60_vector:

        print('rt60', rt60)

        for m in range(M):
            irs[m] = generate_late_reverberation_ir_from_rt60(start_time, rt60, length, fs)
            irs[m, 0] *= y[0, m]

        plt.plot(np.arange(length * fs) / fs, irs[0])
        plt.grid()
        plt.show()

        ## CONVOLVE IRS WITH AUDIO

        audio_file_length = 2. ## seconds
        audio_file_length_samples = int(audio_file_length*fs)

        s_dir = librosa.core.load(af, sr=fs, mono=True)[0][:audio_file_length_samples]

        bformat = np.zeros((M, audio_file_length_samples))
        for m in range(M):
            bformat[m] = scipy.signal.fftconvolve(s_dir, irs[m])[:audio_file_length_samples]  # keep original length


        r=4
        window_size = 256
        window_overlap = window_size//2

        _, _, S_dir = scipy.signal.stft(s_dir, fs, nperseg=window_size, noverlap=window_overlap )


        s_tot_ambi = psa.Signal(bformat, fs, 'acn', 'n3d')
        S_tot_ambi = psa.Stft.fromSignal(s_tot_ambi,
                                window_size=window_size,
                                window_overlap=window_overlap
                                )
        doa = psa.compute_DOA(S_tot_ambi)

        psa.plot_signal(s_tot_ambi)
        psa.plot_magnitude_spectrogram(S_tot_ambi)
        psa.plot_doa(doa)


        ### DIFFUSENESS


        ita = S_tot_ambi.compute_ita(r=r)
        psa.plot_directivity(ita, title='ita')

        # ita_re = S_tot_ambi.compute_ita_re(r=r)
        # psa.plot_directivity(ita_re, title='ita re')

        ### KSI
        ksi = S_tot_ambi.compute_ksi(r=r)
        psa.plot_directivity(ksi, title='ksi')

        # ksi_re = S_tot_ambi.compute_ksi_re(r=r)
        # psa.plot_directivity(ksi_re, title='ksi re')

        # # difference is very small. also ksi always greater than ita
        # psa.plot_directivity(ksi_re-ita_re, title='ksi - ita (diff)')
        #
        # ## decomposition
        msc = S_tot_ambi.compute_msc(r=r)
        psa.plot_directivity(msc, title='msc')

        msc_re = S_tot_ambi.compute_msc_re(r=r)
        psa.plot_directivity(msc_re, title='msc_re')

        msw = S_tot_ambi.compute_msw(r=r)
        psa.plot_directivity(msw, title='normalized velocity')

        A = np.sqrt(msc.data[0] * msw.data[0] + msc.data[1] * msw.data[1] + msc.data[2] * msw.data[2])
        A_stft = psa.Stft(msc.t, msc.f, A, msc.sample_rate)
        psa.plot_directivity(A_stft, title='sqrt(W*MSC)')

        A_re = np.sqrt(msc_re.data[0] * msw.data[0] + msc_re.data[1] * msw.data[1] + msc_re.data[2] * msw.data[2])
        A_re_stft = psa.Stft(msc_re.t, msc_re.f, A_re, msc_re.sample_rate)
        psa.plot_directivity(A_re_stft, title='sqrt(W*MSCre)')
        # plt.show()
        #
        # # Only numerical artifacts
        # assert np.allclose(ksi_re.data[0],A)

plt.show()



# #### Time
#
# T = 10
#
# t0 = time.time()
# for t in range(T):
#     S_tot_ambi.compute_ita(r=r)
# t1 = time.time()
# mean = (t1-t0)/T
# print('ita, ', mean)
#
# t0 = time.time()
# for t in range(T):
#     S_tot_ambi.compute_ksi(r=r)
# t1 = time.time()
# mean = (t1-t0)/T
# print('ksi, ', mean)
