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

import librosa



# %% PARAMETERS

# parameters from "Online Dereverberation..."

fs = 16000
dimM = 4  # first order

window_size = 512 # samples
# window_overlap = window_size//2 # samples, 16 ms at 16k
# hop_size = window_size//2 #
hop_size = window_size #
noverlap = window_size - hop_size
# nfft = window_size
nfft = window_size * 2
window_type = 'boxcar'
# window_type = 'hann'


audio_file_length = 5  ## seconds



ir_type = 'room'


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
# azi = np.pi/2
# incl = np.pi/2

print('AZI - ELE', azi, np.pi/2 - incl)
dirs = np.asarray([[azi, incl]])
basisType = 'real'
y = masp.get_sh(1, dirs, basisType) * np.sqrt(4*np.pi) * [1, 1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)] ## ACN, SN3D

if ir_type is 'simple':

    ir_start_time = window_size / fs
    # ir_start_time = 0.08 # D = 1
    rt60 = 0.075
    # ir_length = 0.064
    ir_length = 0.3


    irs = np.zeros((dimM, int(ir_length * fs)))

    # Generate direct path plus reverberant tail
    for m in range(dimM):
        irs[m] = generate_late_reverberation_ir_from_rt60(ir_start_time, rt60, ir_length, fs)

    # Set direct path gains
    irs[:,0] = y

    ## normalize tail energy to 1
    # tail = irs[0, 1:]
    # tail_energy = np.sum(np.power(np.abs(tail), 2))
    # tail = tail/np.sqrt(tail_energy)
    # irs[:,1:] = irs[:,1:]/np.sqrt(tail_energy)

elif ir_type is 'room':

    room = np.array([10.2, 7.1, 3.2])
    rt60 = np.array([0.5])
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

    maxlim = 1.  # just stop if the echogram goes beyond that time ( or just set it to max(rt60) )
    limits = np.minimum(rt60, maxlim)

    abs_echograms = srs.compute_echograms_sh(room, src, rec, abs_wall, limits, rec_orders)
    irs = srs.render_rirs_sh(abs_echograms, band_centerfreqs, fs).squeeze().T
    # Normalize as SN3D
    irs *= np.sqrt(4 * np.pi)
    irs *= np.asarray([1, 1. / np.sqrt(3), 1. / np.sqrt(3), 1. / np.sqrt(3)])[:,np.newaxis]  ## ACN, SN3D

elif ir_type is 'delta':

    # ir_length = 0.05
    ir_length_samples = window_size * 3
    # ir_length_samples = window_size * 3
    ir_length = ir_length_samples / fs

    irs = np.zeros((dimM, int(ir_length * fs)))
    irs[:,0] = 1
    irs[:,2*window_size] = 1
    irs[:,window_size] = -0.5

    # # Generate direct path plus reverberant tail
    # for m in range(dimM):
    #     irs[m] = generate_late_reverberation_ir_from_rt60(ir_start_time, rt60, ir_length, fs)
    #
    # # Set direct path gains
    # irs[:,0] = y
    # irs[:, 0] = 1
    # irs[:, int((ir_length * fs)//2)] = 1


# %% defs


# https://stackoverflow.com/questions/2459295/invertible-stft-and-istft-in-python
def stft(x, framesamp, hopsamp, nfft, window_type='hann'):

    w = scipy.signal.windows.get_window(window_type, framesamp)
    # s = framesamp/np.sum(w)
    s = hopsamp/np.sum(w)

    # todo ensure nfft >= framesamp
    # next power of 2 to make things easier
    # https://stackoverflow.com/questions/466204/rounding-up-to-next-power-of-2
    nfft = int(np.power(2, np.ceil(np.log(nfft)/np.log(2))))
    K = nfft//2 + 1
    # N = int(np.ceil((len(x)-framesamp)/hopsamp))
    N = int(np.ceil(len(x)/hopsamp))
    X = np.zeros((K, N), dtype='complex')
    for n, i in enumerate(range(0, len(x)-framesamp, hopsamp)):
        print(n, i, i+framesamp)
        y = x[i:i+framesamp] * w * s
        X[:, n] = np.fft.rfft(y, nfft)
    return X / framesamp


def istft(X, framesamp, hopsamp, nfft, window_type='hann'):

    w = scipy.signal.windows.get_window(window_type, framesamp)
    if not scipy.signal.check_COLA(w, framesamp, framesamp-hopsamp):
        raise ValueError('COLA criterium not met')

    N = np.shape(X)[1]
    # s = np.sum(w) / framesamp
    s = np.sum(w) / hopsamp
    print(s)
    x = np.zeros(N*hopsamp+framesamp)
    for n,i in enumerate(range(0, len(x)-framesamp, hopsamp)):
        print(n, i, i + framesamp)
        x[i:i+framesamp] += (np.fft.irfft(X[:,n], nfft)[:framesamp])
    return x * s * framesamp


def ctf(X, H, nfft):
    K_x, N = X.shape
    K_h, L = H.shape
    assert K_x == K_h
    Y = np.zeros(np.shape(X), dtype='complex')
    for n in range(N):
        for l in range(L):
            if n >= l:
                Y[:, n] += X[:, n - l] * H[:, l]
    return Y * nfft


# %% SYNTHESIZE AUDIOS


af = '/Volumes/Dinge/audio/410298__inspectorj__voice-request-26b-algeria-will-rise-again-serious.wav'
audio_file_length_samples = int(audio_file_length * fs)
x = librosa.core.load(af, sr=fs, mono=True)[0][:audio_file_length_samples]
h = irs[0]

true_y = scipy.signal.convolve(x, h)[:audio_file_length_samples]


# sicpy
_, _, true_Y1 = scipy.signal.stft(true_y, fs, window=window_type, nperseg=window_size, noverlap=noverlap, nfft=nfft)
_, _, X1 = scipy.signal.stft(x, fs, window=window_type, nperseg=window_size, noverlap=noverlap, nfft=nfft, boundary=None)
_, _, H1 = scipy.signal.stft(h, fs, window=window_type, nperseg=window_size, noverlap=noverlap, nfft=nfft, boundary=None)
Y1 = ctf(X1, H1, nfft)
K, N = Y1.shape
true_Y1 = true_Y1[:, :N]
_, y1 = scipy.signal.istft(Y1, fs, window=window_type, nperseg=window_size, noverlap=noverlap, nfft=nfft)
y1 = y1[:audio_file_length_samples]

# my
true_Y2 = stft(true_y, window_size, hop_size, nfft, window_type)
X2 = stft(x, window_size, hop_size, nfft, window_type)
H2 = stft(h, window_size, hop_size, nfft, window_type)
Y2 = ctf(X2, H2, nfft)
y2 = istft(Y2, window_size, hop_size, nfft, window_type)
y2 = y2[:audio_file_length_samples]


# framesamp = window_size
# hopsamp = hop_size

#
#



plot_magnitude_spectrogram(X1, 'X1')
plot_magnitude_spectrogram(X2, 'X2')

plot_magnitude_spectrogram(H1, 'H1')
plot_magnitude_spectrogram(H2, 'H2')

plot_magnitude_spectrogram(Y1, 'Y1')
plot_magnitude_spectrogram(Y2, 'Y2')

plot_magnitude_spectrogram(true_Y1, 'true_Y1')
plot_magnitude_spectrogram(true_Y2, 'true_Y2')

plot_signal(x, 'x')
plot_signal(y1, 'y1')
plot_signal(y2, 'y2')
plot_signal(true_y, 'true_y')
#
# plot_magnitude_spectrogram(np.abs(true_Y1-Y1), 'abs of difference')
# plot_magnitude_spectrogram(np.abs(true_Y1)-np.abs(Y1), 'difference of abs')
plt.show()


# _, y1 = scipy.signal.istft(Y1, fs, window=window_type, nperseg=window_size, noverlap=noverlap, nfft=nfft)
# y1 = y1[:audio_file_length_samples]

# plot_signal(x, 'x')
# plot_signal(h, 'h')
# plot_signal(y1, 'y')
# # plot_signal(y2, 'y')
# plot_signal(true_y, 'true_y')
# plt.show()
# play(true_y, fs)
play(true_y, fs)
play(y2, fs)

# _, y1 = scipy.signal.istft(Y1, fs, window=window_type, nperseg=window_size, noverlap=noverlap, nfft=nfft)
# plt.show()

