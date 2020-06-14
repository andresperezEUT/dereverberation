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

audio_file_length = 3.  ## seconds

## MAR-------
# D = int(np.floor(ir_start_time * fs / window_overlap)) # ir start frame
D = 1
print('D=',D)
if D < 1:
    raise Warning('D should be at least 1!!!')
L = 5  # total number of frames for the filter
Lc = dimM * dimM * L

alpha = 0.4
ita = np.power(10, -35 / 10)
alpha_RLS = 0.99

noise_power = 1e-5

ir_type = 'simple'


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

    ir_start_time = (window_size-window_overlap) * D / fs
    # ir_start_time = 0.08 # D = 1
    rt60 = 0.5
    # ir_length = 0.064
    ir_length = (window_size-window_overlap) * (D+L) / fs

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

mono_s_t = librosa.core.load(af, sr=fs, mono=True)[0][2*audio_file_length_samples:3*audio_file_length_samples]
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


## PLOT STUFF

plt.plot(mono_s_t)
plot_signal(s_t, title='s_t')
plot_magnitude_spectrogram(s_tf, title='s_tf')
plot_signal(irs, title='irs')
plot_signal(y_t, title='y_t')
plot_magnitude_spectrogram(y_tf, title='y_tf')
plt.show()




# %% ANALYSIS

est_s1_tf, est_c1_tf = dereverberation_MAR(y_tf, D, L, alpha, ita)
_, est_s1_t = scipy.signal.istft(est_s1_tf, fs, window=window_type, nperseg=window_size, noverlap=window_overlap, nfft=nfft)
est_s1_t = est_s1_t[:, :audio_file_length_samples]

# est_s2_tf, est_c2_tf = dereverberation_RLS(y_tf, D, L, alpha, alpha_RLS)
# _, est_s2_t = scipy.signal.istft(est_s2_tf, fs, window=window_type, nperseg=window_size, noverlap=window_overlap, nfft=nfft)
# est_s2_t2 = est_s2_t[:,:audio_file_length_samples]
#
# est_s3_tf, est_c3_tf = dereverberation_MAR_oracle(y_tf, D, L, alpha, ita, s_tf)
# _, est_s3_t = scipy.signal.istft(est_s3_tf, fs, window=window_type, nperseg=window_size, noverlap=window_overlap, nfft=nfft)
# est_s3_t = est_s3_t[:,:audio_file_length_samples]

# PLOT

plt.figure()
plot_magnitude_spectrogram(s_tf[0], 's 0')
plt.show()

plt.figure()
plot_magnitude_spectrogram(y_tf[0], 'y 0')
plt.show()

plt.figure()
plot_magnitude_spectrogram(est_s1_tf[0], 'est_s_tf 0')
plt.show()

# plt.figure()
# plot_magnitude_spectrogram(est_s2_tf[0], 'est_s2_tf 0')
# plt.show()
#
# plt.figure()
# plot_magnitude_spectrogram(est_s3_tf[0], 'est_s3_tf 0')
# plt.show()





# %% RECONSTRUCTION

## TEST RECONSTRUCTION
C1 = unwrap_MAR_coefs(est_c1_tf, L)

output1_tf = apply_MAR_coefs(C1, est_s1_tf, L, D)
_, output1_t = scipy.signal.istft(output1_tf, fs, window=window_type, nperseg=window_size, noverlap=window_overlap, nfft=nfft)
output1_t = output1_t[:,:audio_file_length_samples]

plt.figure()
plot_magnitude_spectrogram(output1_tf[0], 'output1_tf 0')
plt.show()

## Difference with true output
diff1 = y_tf - output1_tf
plt.figure()
plot_magnitude_spectrogram(diff1, title='diff1')
plt.show()


#
# C2 = unwrap_MAR_coefs(est_c2_tf, L)
#
# output2_tf = apply_MAR_coefs(C2, est_s2_tf, L, D)
# _, output2_t = scipy.signal.istft(output2_tf, fs, window=window_type, nperseg=window_size, noverlap=window_overlap, nfft=nfft)
# output2_t = output2_t[:,:audio_file_length_samples]
#
# plt.figure()
# plot_magnitude_spectrogram(output2_tf[0], 'output2_tf 0')
# plt.show()
#
# ## Difference with true output
# diff2 = y_tf - output2_tf
# plt.figure()
# plot_magnitude_spectrogram(diff2, title='diff2')
# plt.show()
#
#
# C3 = unwrap_MAR_coefs(est_c3_tf, L)
#
# output3_tf = apply_MAR_coefs(C3, est_s3_tf, L, D)
# _, output3_t = scipy.signal.istft(output3_tf, fs, window=window_type, nperseg=window_size, noverlap=window_overlap, nfft=nfft)
# output3_t = output3_t[:,:audio_file_length_samples]
#
# plt.figure()
# plot_magnitude_spectrogram(output3_tf[0], 'output3_tf 0')
# plt.show()
#
# ## Difference with true output
# diff3 = y_tf - output3_tf
# plt.figure()
# plot_magnitude_spectrogram(diff3, title='diff3')
# plt.show()







#
# # ## reconstruction with averaged coefs
# r_tf = apply_MAR_coefs(C, est_s_tf, L, D, time_average=True)
# _, r_t = scipy.signal.istft(r_tf, fs, window=window_type, nperseg=window_size, noverlap=window_overlap, nfft=nfft)
# r_t = r_t[:,:audio_file_length_samples]
#
#
# plt.figure()
# plot_magnitude_spectrogram(r_tf[0], title='averaged C reconstruction')
# plt.show()
# #

# play(s_t[0], fs)
# play(est_s1_t[0], fs)
# play(y_t[0], fs)
# play(output1_t[0], fs)

# ## Difference with true output
# diff1 = r_tf - output_tf
# plt.figure()
# plot_magnitude_spectrogram(diff1[0], title='diff true C vs averaged C')
# plt.show()
#
#
#






# %% test with another signal

# # Open new signal
# af2 = audio_files[1]
# # af2 = '/Volumes/Dinge/audio/410298__inspectorj__voice-request-26b-algeria-will-rise-again-serious.wav'
#
# mono_s2_t = librosa.core.load(af2, sr=fs, mono=True)[0][3*audio_file_length_samples:4*audio_file_length_samples]
# s2_t = mono_s2_t * y.T  # dry ambisonic target
# f, t, s2_tf = scipy.signal.stft(s2_t, fs, window=window_type, nperseg=window_size, noverlap=window_overlap, nfft=nfft)
#
#
# #### Re-reverberate
#
# # 1) Full matrix
# r1_tf = apply_MAR_coefs(C1, s2_tf, L, D)
# _, r1_t = scipy.signal.istft(r1_tf, fs, window=window_type, nperseg=window_size, noverlap=window_overlap, nfft=nfft)
# r1_t = r1_t[:, :audio_file_length_samples]
#
# # 2) time-averaged matrix
# r2_tf = apply_MAR_coefs(C1, s2_tf, L, D, time_average=True)
# _, r2_t = scipy.signal.istft(r2_tf, fs, window=window_type, nperseg=window_size, noverlap=window_overlap, nfft=nfft)
# r2_t = r2_t[:,:audio_file_length_samples]
#
# plt.figure()
# plot_magnitude_spectrogram(s2_tf[0], title='s2_tf')
# plt.show()
#
# plt.figure()
# plot_magnitude_spectrogram(r1_tf[0], title='r1_tf')
# plt.show()
#
# plt.figure()
# plot_magnitude_spectrogram(r2_tf[0], title='r2_tf')
# plt.show()



# PLAY
# play(s_t[0],fs)
# play(s2_t[0],fs)
# play(r1_t[0],fs)
# play(r2_t[0],fs)





# %% transition matrix


# e = get_MAR_transition_matrix_eigenvalues(C)
#
# plt.figure()
# plt.title('1st eigenvalue magnitude')
# plt.pcolormesh(np.abs(e[0]), cmap='inferno')
# plt.colorbar()
# plt.show()
#
# plt.figure()
# plt.title('unstable filters (mag>1) ')
# plt.pcolormesh((np.abs(e[0, :, :]) > 1).astype(int), cmap='inferno')
# plt.colorbar()
# plt.show()
#
#
# # time average
# e2 = get_MAR_transition_matrix_eigenvalues(C, time_average=True)
#
# plt.figure()
# plt.title('1st eigenvalue magnitude')
# plt.plot(np.abs(e2[0])) # greatest eigenvalue
#
# plt.figure()
# plt.title('unstable filters (mag>1) ')
# plt.plot((np.abs(e2[0])>1).astype(int)) # greatest eigenvalue
#
# plt.show()







# %% COEFFICIENTS
# Average over time
# plt.style.use('seaborn-whitegrid')


# C = C1
# c = np.mean(C[:,:,:,:,D+L:], axis=-1) # over time
# c_std = np.std(C[:,:,:,:,D+L:], axis=-1) # over time
#
#
# # get 16 colors
# cmap = plt.get_cmap("tab20c")
# plt.style.use('seaborn-whitegrid')
#
# k = dimK//2
# plt.figure()
# for m in range(dimM):
#     plt.errorbar(range(L),
#                  np.abs(c[:,m,m,k]),
#                  yerr=c_std[:,m,m,k],
#                  fmt='o-',
#                  markersize=5,
#                  capsize=3,
#                  label=str(m) + str(m),
#                  # c=cmap.colors[m * dimM + m]
#                  )
# plt.legend(loc='upper right')
# plt.show()
#
# plt.figure()
# for m in range(dimM):
#     plt.plot(np.abs(irs[m,window_size:]),
#              label=str(m))
# plt.legend(loc='upper right')
# plt.show()


############
# #
# k = dimK//2
# plt.figure()
# # plt.grid()
# plt.title('average over time, k='+str(k))
# for m1 in range(dimM):
#     for m2 in range(dimM):
#         plt.errorbar(range(L),
#                      np.abs(c[:,m1,m2,k]),
#                      yerr=c_std[:,m1,m2,k],
#                      fmt='o-',
#                      markersize=5,
#                      capsize=3,
#                      label=str(m1)+str(m2),
#                      # cmap=cmap
#                      c=cmap.colors[m1*dimM+m2]
#                      )
# plt.legend(loc='upper right')
# plt.show()



############
#
# k = dimK//2
# plt.figure()
# # plt.grid()
# plt.title('average over time, k='+str(k))
# for m1 in range(dimM):
#     for m2 in range(dimM):
#         plt.errorbar(range(L),
#                      np.abs(c[:,m1,m2,k]),
#                      yerr=c_std[:,m1,m2,k],
#                      fmt='o-',
#                      markersize=5,
#                      capsize=3,
#                      label=str(m1)+str(m2),
#                      # cmap=cmap
#                      c=cmap.colors[m1*dimM+m2]
#                      )
# plt.legend(loc='upper right')
# plt.show()

# C = C1
# plt.figure()
# plt.title('Abs value of mean C accross time')
# for k in np.arange(0, dimK, 50):
#
#     c_mean = np.mean(C[:,:,:,k,:], axis=(1,2,3))
#     c_std = np.std(C[:,:,:,k,:], axis=(1,2,3))
#     plt.errorbar(range(L),
#                  np.abs(c_mean),
#                  # yerr=c_std,
#                  fmt='o-',
#                  markersize=3,
#                  # capsize=3,
#                  label='k='+str(k),
#                  )
#
# plt.plot( np.abs(np.mean(C[:,:,:,:,:], axis=(1,2,3,4))), label='sum')
# plt.legend(loc='upper right')
# plt.show()






# plt.figure()
# plt.title('Determinant of mean C accross time')
# for k in np.arange(0, dimK, 50):
#
#     c_det = np.linalg.det(c[:,:,:,k])
#     # c_std = np.std(C[:,:,:,k,:], axis=(1,2,3))
#     plt.errorbar(range(L),
#                  np.abs(c_det),
#                  # yerr=c_std,
#                  fmt='o-',
#                  markersize=3,
#                  # capsize=3,
#                  label='k='+str(k),
#                  )
# plt.legend(loc='upper right')
# plt.show()
#

#
# # %% Direction of arrivals
#
# def plot_doa(x_t, title):
#     x = psa.Signal(x_t, fs, 'acn', 'sn3d')
#     X = psa.Stft.fromSignal(x, window_size=window_size, window_overlap=window_overlap, nfft=nfft)
#     X_doa = psa.compute_DOA(X)
#     psa.plot_doa(X_doa, title)
#     plt.show()
#     return X_doa
#
# # True reverberant
# Y_doa = plot_doa(y_t, 'y_t')
# # Estimated reverberant
# O_doa = plot_doa(output_t, 'output_t')
# # Difference
# psa.plot_doa(Y_doa-O_doa, 'difference true-estimated')
# plt.show()
#
#
# # Averaged reverberant
# R_doa = plot_doa(r_t, 'r_t')
# # Difference
# psa.plot_doa(Y_doa-R_doa, 'difference true-averaged')
# plt.show()
# psa.plot_doa(O_doa-R_doa, 'difference estimated-averaged')
# plt.show()
#
#
# # Resynthesis signal
# R1_doa = plot_doa(r1_t, 'r1_t')
# R2_doa = plot_doa(r2_t, 'r2_t')
#
#
#
#
# # %% Write files
#
# import soundfile as sf
#
# output_folder = '/Volumes/Dinge/audio/output_rereverb'
# def write_file(data, name, path):
#     name += '.wav'
#     sf.write(os.path.join(output_folder, name), data.T, fs)
#
# write_file(s_t, 's_t', output_folder)
# write_file(est_s_t, 'est_s_t', output_folder)
# write_file(y_t, 'y_t', output_folder)
# write_file(output_t, 'output_t', output_folder)
# write_file(r_t, 'r_t', output_folder)
# write_file(s2_t, 's2_speech_t', output_folder)
# write_file(r1_t, 'r1_speech_t', output_folder)
# write_file(r2_t, 'r2_speech_t', output_folder)