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
window_overlap = window_size//2
# window_overlap = 0
nfft = window_size
window_type = 'hann'

audio_file_length = 4  ## seconds

## MAR-------
# D = int(np.floor(ir_start_time * fs / window_overlap)) # ir start frame
tau = 1
print('tau=',tau)
if tau < 1:
    raise Warning('D should be at least 1!!!')
L = 10  # total number of frames for the filter
Lc = dimM * dimM * L

alpha = 0.4
ita = np.power(10, -35 / 10)
alpha_RLS = 0.99

noise_power = 1e-5

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
# azi = 0
# incl = np.pi/2

print('AZI - ELE', azi, np.pi/2 - incl)
dirs = np.asarray([[azi, incl]])
basisType = 'real'
y = masp.get_sh(1, dirs, basisType) * np.sqrt(4*np.pi) * [1, 1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)] ## ACN, SN3D

if ir_type is 'simple':

    ir_start_time = (window_size-window_overlap) * 3 / fs
    # ir_start_time = 0.08 # D = 1
    rt60 = 0.5
    # ir_length = 0.064
    ir_length = (window_size-window_overlap) * (tau+L//2) / fs

    irs = np.zeros((dimM, int(ir_length * fs)))

    # Generate direct path plus reverberant tail
    for m in range(dimM):
        irs[m] = generate_late_reverberation_ir_from_rt60(ir_start_time, rt60, ir_length, fs)

    # Set direct path gains
    irs[:,0] = y

    ## normalize tail energy to 1
    tail = irs[0, 1:]
    tail_energy = np.sum(np.power(np.abs(tail), 2))
    # tail = tail/np.sqrt(tail_energy)
    irs[:,1:] = irs[:,1:]/np.sqrt(tail_energy)

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



# %% SYNTHESIZE AUDIOS

# # af = audio_files[1]
af = '/Volumes/Dinge/audio/410298__inspectorj__voice-request-26b-algeria-will-rise-again-serious.wav'
#
# # Open audio files and encode into ambisonics
audio_file_length_samples = int(audio_file_length * fs)

#

mono_s_t = librosa.core.load(af, sr=fs, mono=True)[0][audio_file_length_samples:2*audio_file_length_samples]
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


# estimate
est_s_tf, C, est_phi = estimate_MAR_sparse_parallel(y_tf, L, tau, p, i_max, ita, epsilon)
_, est_s_t = scipy.signal.istft(est_s_tf, fs, window=window_type, nperseg=window_size, noverlap=window_overlap, nfft=nfft)
est_s_t = est_s_t[:, :audio_file_length_samples]

# # oracle estimate
# est_s_oracle_tf, C_oracle = estimate_MAR_sparse_oracle(y_tf, s_tf, L, tau, p, i_max, ita, epsilon)
# _, est_s_oracle_t = scipy.signal.istft(est_s_oracle_tf, fs, window=window_type, nperseg=window_size, noverlap=window_overlap, nfft=nfft)
# est_s_oracle_t = est_s_oracle_t[:, :audio_file_length_samples]
#
# # identity estimate
# est_s_identity_tf, C_identity = estimate_MAR_sparse_identity(y_tf, L, tau, p, i_max, ita, epsilon)
# _, est_s_identity_t = scipy.signal.istft(est_s_identity_tf, fs, window=window_type, nperseg=window_size, noverlap=window_overlap, nfft=nfft)
# est_s_identity_t = est_s_identity_t[:, :audio_file_length_samples]


# %% Plot spectrograms

# dry signal
plot_magnitude_spectrogram(s_tf[0], 's_tf')
# true reverberant signal
plot_magnitude_spectrogram(y_tf[0], 'y_tf')
# estimated dry signal
plot_magnitude_spectrogram(est_s_tf[0], 'est_s_tf')
# estimated oracle dry signal
# plot_magnitude_spectrogram(est_s_oracle_tf[0], 'est_s_oracle_tf')
# estimated identity dry signal
# plot_magnitude_spectrogram(est_s_identity_tf[0], 'est_s_identity_tf')

plt.show()


# %% Plot statistics


# plt.plot(irs[0])
# plt.show()
#
# plot_magnitude_spectrogram(s_tf, 's_tf')
# plot_magnitude_spectrogram(y_tf, 'y_tf')
# plot_magnitude_spectrogram(est_s_tf, 'est_s_tf')
# plt.show()
#
k = dimK//2+1
for m in range(dimM):
    plt.plot(np.abs(C[k,:,m]), label=str(m))
plt.legend()
plt.show()

plt.figure()
plt.title('rt60='+str(rt60))
for m in range(dimM):
    plt.plot(np.mean(np.abs(C[:,:,m]),axis=0), label=str(m))
plt.legend()
plt.show()

# # estimated phi
# plt.title('estimated scm')
# norm_phi = phi[k, :, :]/np.max(phi[k, :, :])
# plt.imshow( np.abs(norm_phi), cmap='magma')
# plt.colorbar()
# plt.grid()
# plt.show()
#
# true phi
plt.figure()
plt.title('true scm')
s = s_tf[:, k, :].T
true_phi = herm(s) @ s
norm_true_phi = true_phi / np.max(true_phi)
plt.imshow( np.abs(norm_true_phi), cmap='magma')
plt.colorbar()
plt.grid()
plt.show()

# play(s_t[0], fs)
# play(y_t[0], fs)
# play(est_s_t[0], fs)




# %% RECONSTRUCTION

x2_tf = np.zeros((dimM, dimK, dimN), dtype=complex)
for k in range(dimK):
    for n in range(tau, dimN):

        x = np.zeros(L*dimM, dtype=complex)
        for m in range(dimM):
            for l in range(L):
                x[L*m+l] = x2_tf[m, k, n-tau-l]

        x2_tf[:,k,n] = est_s_tf[:,k,n] + x @ C[k,:,:]


# plot_magnitude_spectrogram(s_tf)
plot_magnitude_spectrogram(x2_tf, 'x2_tf')
# plot_magnitude_spectrogram(y_tf)
# plot_magnitude_spectrogram(est_s_tf)
# difference
plot_magnitude_spectrogram(np.abs(y_tf)-np.abs(x2_tf))
plt.show()

# play(s_t[0], fs)
# play(est_s_t[0], fs)
# play(y_t[0], fs)

# %% RE-RECONSTRUCTION

# Open new signal
af2 = audio_files[1]
# af2 = '/Volumes/Dinge/audio/410298__inspectorj__voice-request-26b-algeria-will-rise-again-serious.wav'

mono_s2_t = librosa.core.load(af2, sr=fs, mono=True)[0][audio_file_length_samples:2*audio_file_length_samples]
s2_t = mono_s2_t * y.T  # dry ambisonic target
f, t, s2_tf = scipy.signal.stft(s2_t, fs, window=window_type, nperseg=window_size, noverlap=window_overlap, nfft=nfft)

dimM, dimK, dimN = s2_tf.shape

#### Re-reverberate

x2_tf = np.zeros((dimM, dimK, dimN), dtype=complex)
for k in range(dimK):
    for n in range(tau, dimN):
        x = np.zeros(L*dimM, dtype=complex)
        for m in range(dimM):
            for l in range(L):
                x[L*m+l] = x2_tf[m, k, n-tau-l]

        x2_tf[:,k,n] = s2_tf[:,k,n] + x @ C[k,:,:]

_, x2_t = scipy.signal.istft(x2_tf, fs, window=window_type, nperseg=window_size, noverlap=window_overlap, nfft=nfft)
x2_t = x2_t[:, :audio_file_length_samples]




# Signal with Real IRS
y2_t = np.zeros((dimM, audio_file_length_samples))  # reverberant signal
for m in range(dimM):
    y2_t[m] = scipy.signal.fftconvolve(mono_s2_t, irs[m])[:audio_file_length_samples]  # keep original length
_, _, y2_tf = scipy.signal.stft(y2_t, fs, window=window_type, nperseg=window_size, noverlap=window_overlap, nfft=nfft)

plot_magnitude_spectrogram(s2_tf[0], title='s2_tf')
plot_magnitude_spectrogram(x2_tf[0], 'x2_tf')
plot_magnitude_spectrogram(y2_tf[0], 'y2_tf')
plt.show()


# play(s2_t[0], fs)
# play(x2_t[0], fs)
# play(y2_t[0], fs)




# %% STABILITY

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
plt.xlabel('k')
plt.ylabel(r'$|\lambda_0|$')
plt.show()

# play(s2_t[0], fs)
# play(x2_t[0], fs)


# %% Direction of arrivals

def plot_doa(x_t, title):
    x = psa.Signal(x_t, fs, 'acn', 'sn3d')
    X = psa.Stft.fromSignal(x, window_size=window_size, window_overlap=window_overlap, nfft=nfft)
    X_doa = psa.compute_DOA(X)
    psa.plot_doa(X_doa, title)
    plt.show()
    return X_doa

# True reverberant
Y_doa = plot_doa(y_t, 'y_t')
# Estimated reverberant
O_doa = plot_doa(est_s_t, 'output_t')
# Difference
psa.plot_doa(Y_doa-O_doa, 'difference true-estimated')
plt.show()


# Reverberant

# Difference
psa.plot_doa(y2_t, 'y2')
plt.show()
psa.plot_doa(x2_t, 'x2')
plt.show()

#
# # Resynthesis signal
# R1_doa = plot_doa(r1_t, 'r1_t')
# R2_doa = plot_doa(r2_t, 'r2_t')
#
#