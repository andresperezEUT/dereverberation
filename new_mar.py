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


#### parameters from "Online Dereverberation..."

dimM = 4  # first order

ir_start_time = 0.016 # D=1
# ir_start_time = 0.08 # D = 1
rt60 = 0.1
ir_length = 0.096
fs = 16000


window_size = 512 # samples
window_overlap = window_size//2 # samples, 16 ms at 16k
nfft = window_size * 2
D = int(np.floor(ir_start_time * fs / window_overlap)) # ir start frame
print('D=',D)
if D < 1:
    raise Warning('D should be at least 1!!!')

# audio_file_length = 2.  ## seconds
audio_file_length = 5.  ## seconds

## MAR-------
Lar = 9 # index of last fame
assert Lar >= D
L = Lar-D+1 # total number of frames for the MAR filter
Lc = dimM * dimM * L

alpha = 0.4
ita = np.power(10, -35 / 10)
alpha_RLS = 0.99



# TODO: real uniform along the sphere
# azi = np.random.rand()*2*np.pi
# incl = np.random.rand()*np.pi

azi = 0
incl = np.pi/2

print('AZI - ELE', azi, np.pi/2 - incl)
dirs = np.asarray([[azi, incl]])
basisType = 'real'
y = masp.get_sh(1, dirs, basisType) * np.sqrt(4*np.pi) * [1, 1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)] ## ACN, SN3D

## GET FOUR RANDOM IRS, ONE FOR EACH CHANNEL, AND APPLY GAINS TO THE DIRECT PATH
## the late reverb should keep the same amplitude (energy) regardless of the direction

irs = np.zeros((dimM, int(ir_length*fs)))

#### get audio
audio_files = []
data_folder_path = '/Volumes/Dinge/DSD100subset/Sources'
for root, dir, files in os.walk(data_folder_path):
    for f in files:
        extension = os.path.splitext(f)[1]
        if 'wav' in extension:
            audio_files.append(os.path.join(root, f))




################################################################
################################################################
################################################################

af = audio_files[1]
# af = '/Volumes/Dinge/audio/410298__inspectorj__voice-request-26b-algeria-will-rise-again-serious.wav'

for m in range(dimM):
    irs[m] = generate_late_reverberation_ir_from_rt60(ir_start_time, rt60, ir_length, fs)

for s in range(int(ir_length*fs)):
    for m in range(dimM):
        irs[m,s] += np.random.normal()*1e-8

## normalize tail energy to 1
# tail = irs[0, 1:]
# tail_energy = np.sum(np.power(np.abs(tail), 2))
# tail = tail/np.sqrt(tail_energy)
# irs[:,1:] = irs[:,1:]/np.sqrt(tail_energy)

# Open audio files and encode into ambisonics
audio_file_length_samples = int(audio_file_length * fs)
mono_s_t = librosa.core.load(af, sr=fs, mono=True)[0][audio_file_length_samples:2*audio_file_length_samples]
s_t = mono_s_t * y.T  # dry ambisonic target
f, t, s_tf = scipy.signal.stft(s_t, fs, window='hann', nperseg=window_size, noverlap=window_overlap, nfft=nfft)

# Get reverberant signal
late_irs = copy.copy(irs)
late_irs[:,0] = 0 # remove the delta
r_t = np.zeros((dimM, audio_file_length_samples))  # reverberant signal
for m in range(dimM):
    r_t[m] = scipy.signal.fftconvolve(mono_s_t, late_irs[m])[:audio_file_length_samples]  # keep original length
    # r_t[m] = scipy.signal.fftconvolve(s_t[m], late_irs[m])[:audio_file_length_samples]  # TODO REMOVE:
f, t, r_tf = scipy.signal.stft(r_t, fs, window='hann', nperseg=window_size, noverlap=window_overlap, nfft=nfft)

# Recorded signal is the addition of dry ambisonics and reverberant signal (assuming delta and no early reflections)

y_t = s_t + r_t
# y_t = np.zeros((dimM, audio_file_length_samples))
# for m in range(dimM):
#     y_t[m] = scipy.signal.fftconvolve(mono_s_t, irs[m])[:audio_file_length_samples]  # keep original length
#
f, t, y_tf = scipy.signal.stft(y_t, fs, window='hann', nperseg=window_size, noverlap=window_overlap, nfft=nfft)
# y_stft shape is (M, K, N)
_, dimK, dimN = y_tf.shape


## PLOT STUFF

plt.plot(mono_s_t)
plot_signal(s_t, title='s_t')
plot_magnitude_spectrogram(s_tf, fs, window_overlap, title='s_tf')
plot_signal(irs, title='irs')
plot_signal(late_irs, title='late_irs')
plot_signal(r_t, title='r_t')
plot_magnitude_spectrogram(r_tf, fs, window_overlap, title='r_tf')
plot_signal(y_t, title='y_t')
plot_magnitude_spectrogram(y_tf, fs, window_overlap, title='y_tf')
plt.show()


################################


# est_s_tf = np.empty((dimM, dimK, dimN), dtype='complex')
# e_tf = np.empty((dimM, dimK, dimN), dtype='complex')
# est_c_tf = np.empty((Lc, dimK, dimN), dtype='complex')

# C_matrix = np.empty(((Lar-D+1), dimM, dimM, dimK, dimN), dtype='complex') # t is ordered from Lar to D

#######################################
est_s_tf, est_c_tf = dereverberation_MAR(y_tf, D, L, alpha, ita)


_, est_s_t = scipy.signal.istft(est_s_tf, fs, nperseg=window_size, noverlap=window_overlap, nfft=nfft)
est_s_t = est_s_t[:, :audio_file_length_samples]


# est_s_tf2, est_c_tf2 = dereverberation_RLS(y_tf, dimM, dimN, dimK, Lc, Lar, D, alpha, alpha_RLS)
# est_s_tf3, est_c_tf3 = dereverberation_MAR_oracle(y_tf, dimM, dimN, dimK, Lc, Lar, D, alpha, ita, s_tf)
#######################################



plt.figure()
plt.title('s 0')
librosa.display.specshow(librosa.amplitude_to_db(s_tf[0]), y_axis='linear', x_axis='time', sr=fs, hop_length=window_overlap)
plt.colorbar()
plt.show()

plt.figure()
plt.title('y 0')
librosa.display.specshow(librosa.amplitude_to_db(y_tf[0]), y_axis='linear', x_axis='time', sr=fs, hop_length=window_overlap)
plt.colorbar()
plt.show()

plt.figure()
plt.title('est_s_tf MAR')
librosa.display.specshow(librosa.amplitude_to_db(est_s_tf[0, :, :]), y_axis='linear', x_axis='time', sr=fs, hop_length=window_overlap)
plt.colorbar()
plt.show()
#
# plt.figure()
# plt.title('est_s_tf RLS')
# librosa.display.specshow(librosa.amplitude_to_db(est_s_tf2[0, :, :]), y_axis='linear', x_axis='time', sr=fs, hop_length=window_overlap)
# plt.colorbar()
# plt.show()
#
#
# plt.figure()
# plt.title('est_s_tf oracle')
# librosa.display.specshow(librosa.amplitude_to_db(est_s_tf3[0, :, :]), y_axis='linear', x_axis='time', sr=fs, hop_length=window_overlap)
# plt.colorbar()
# plt.show()



#
# play(s_t[0], fs)
# play(y_t[0], fs)
# play(est_s_t[0], fs)


# _, est_s_t2 = scipy.signal.istft(est_s_tf2, fs, nperseg=window_size, noverlap=window_overlap, nfft=nfft)
# est_s_t2 = est_s_t2[:,:audio_file_length_samples]
# play(est_s_t2[0], fs)
# _, est_s_t3 = scipy.signal.istft(est_s_tf3, fs, nperseg=window_size, noverlap=window_overlap, nfft=nfft)
# est_s_t3 = est_s_t3[:,:audio_file_length_samples]
# play(est_s_t3[0], fs)


################################################
################################################

### TEST RECONSTRUCTION
C = unwrap_MAR_coefs(est_c_tf, L)

output_tf = apply_MAR_coefs(C, est_s_tf, L, D)
_, output_t = scipy.signal.istft(output_tf, fs, nperseg=window_size, noverlap=window_overlap, nfft=nfft)
output_t = output_t[:,:audio_file_length_samples]
play(output_t[0], fs)
play(y_t[0], fs)

plt.title('output_tf')
librosa.display.specshow(librosa.amplitude_to_db(output_tf[0]), y_axis='linear', x_axis='time', sr=fs, hop_length=window_overlap)
plt.colorbar()
plt.show()

## Difference with true output
diff = y_tf - output_tf
plt.figure()
plot_magnitude_spectrogram(diff, fs, window_overlap, title='diff')
plt.show()



################################################################
#
# # ### all n for a given freq
# k = 100
# for n in range(dimN):
#     plt.figure()
#     plt.title('k='+str(k)+' n='+str(n))
#     for m in range(dimM):
#         plt.plot(np.abs(C[:, m, m, k, n]))
#         plt.ylim(0,1)
#     plt.show()
#


# Average over frequency
# n = dimN//2
# for n in range(dimN):
#     plt.figure()
#     plt.title('average over freqs, n='+str(n))
#     for m in range(dimM):
#         plt.errorbar(range(L),
#                      np.abs(np.mean(C[:,m,m,:,n], axis=-1)),
#                      yerr=np.abs(np.std(C[:,m,m,:,n], axis=-1)))
#         # plt.plot(np.mean(np.abs(C[:,m,m,:,n]), axis=-1))

# # Average over time
# k = dimK//2
# plt.figure()
# plt.title('average over time, k='+str(k))
# for m in range(dimM):
#     plt.errorbar(range(L),
#                  np.abs(np.mean(C[:,m,m,k,:], axis=-1)),
#                  yerr=np.abs(np.std(C[:,m,m,k,:], axis=-1)))
# plt.show()

# Total average
# plt.figure()
# plt.title('total average')
# for m in range(dimM):
#     plt.errorbar(range(L),
#                  np.abs(np.mean(C[:,m,m,:,:], axis=(1,2))),
#                  # yerr=np.abs(np.std(C[:,m,m,:,:], axis=(1,2)))
#                  )
#     # plt.errorbar(range(L),np.mean(np.abs(C[:,m,m,:,:]), axis=(1,2)),
#     #              yerr=np.std(np.abs(C[:,m,m,:,:]), axis=(1,2)))
#     # plt.plot(np.mean(np.abs(C[:,m,m,:,:]), axis=(1,2)))
#
# plt.show()

##
import time

e = get_MAR_transition_matrix_eigenvalues(C)


plt.figure()
plt.title('1st eigenvalue magnitude')
librosa.display.specshow(np.abs(e[0]), y_axis='linear', x_axis='time', sr=fs, hop_length=window_overlap) # greatest eigenvalue
plt.colorbar()

plt.figure()

plt.figure()
plt.title('unstable filters (mag>1) ')
librosa.display.specshow((np.abs(e[0,:,:])>1).astype(int), y_axis='linear', x_axis='time', sr=fs, hop_length=window_overlap) # greatest eigenvalue
plt.colorbar()

plt.show()



################################################################
# ## check values
# # k=400
# # n=85
# # np.abs(e[0,k,n])
#
#
# ## reconstruction
r_tf = apply_MAR_coefs(C, est_s_tf, L, D, time_average=True)
_, r_t = scipy.signal.istft(r_tf, fs, nperseg=window_size, noverlap=window_overlap, nfft=nfft)
r_t = r_t[:,:audio_file_length_samples]


plt.title('r_tf')
librosa.display.specshow(librosa.amplitude_to_db(r_tf[0]), y_axis='linear', x_axis='time', sr=fs, hop_length=window_overlap)
plt.colorbar()
plt.show()



play(est_s_t[0], fs)
play(output_t[0], fs)
play(r_t[0], fs)
play(y_t[0], fs)

## Difference with true output
diff1 = r_tf - output_tf
plt.figure()
plot_magnitude_spectrogram(diff1, fs, window_overlap, title='diff full reverb vs mean reverb')
plt.show()



################################################################
# Test with another signal

# Open new signal
# af2 = audio_files[1]
af2 = '/Volumes/Dinge/audio/410298__inspectorj__voice-request-26b-algeria-will-rise-again-serious.wav'

mono_s2_t = librosa.core.load(af2, sr=fs, mono=True)[0][audio_file_length_samples:2*audio_file_length_samples]
s2_t = mono_s2_t * y.T  # dry ambisonic target
f, t, s2_tf = scipy.signal.stft(s2_t, fs, window='hann', nperseg=window_size, noverlap=window_overlap, nfft=nfft)


##### Re-reverberate

# 1) Full matrix
r1_tf = apply_MAR_coefs(C, s2_tf, L, D)
_, r1_t = scipy.signal.istft(r1_tf, fs, nperseg=window_size, noverlap=window_overlap, nfft=nfft)
r1_t = r1_t[:, :audio_file_length_samples]

# 2) time-averaged matrix
r2_tf = apply_MAR_coefs(C, s2_tf, L, D, time_average=True)
_, r2_t = scipy.signal.istft(r2_tf, fs, nperseg=window_size, noverlap=window_overlap, nfft=nfft)
r2_t = r2_t[:,:audio_file_length_samples]

plt.figure()
plt.title('s2_tf')
librosa.display.specshow(librosa.amplitude_to_db(s2_tf[0]), y_axis='linear', x_axis='time', sr=fs, hop_length=window_overlap)
plt.colorbar()
plt.show()

plt.figure()
plt.title('r1_tf')
librosa.display.specshow(librosa.amplitude_to_db(r1_tf[0]), y_axis='linear', x_axis='time', sr=fs, hop_length=window_overlap)
plt.colorbar()
plt.show()

plt.figure()
plt.title('r2_tf')
librosa.display.specshow(librosa.amplitude_to_db(r2_tf[0]), y_axis='linear', x_axis='time', sr=fs, hop_length=window_overlap)
plt.colorbar()
plt.show()


####
c = np.mean(C, axis=-1) # over time
c_std = np.std(C, axis=-1) # over time

e2 = get_MAR_transition_matrix_eigenvalues(C, time_average=True)

plt.figure()
plt.title('1st eigenvalue magnitude')
plt.plot(np.abs(e2[0])) # greatest eigenvalue

plt.figure()
plt.title('unstable filters (mag>1) ')
plt.plot((np.abs(e2[0])>1).astype(int)) # greatest eigenvalue

plt.show()

#
play(s_t[0],fs)
play(s2_t[0],fs)
play(r1_t[0],fs)
play(r2_t[0],fs)

# Plot coefficients
# Average over time
# plt.style.use('seaborn-whitegrid')

# get 16 colors
cmap = plt.get_cmap("tab20c")
plt.style.use('seaborn-whitegrid')
k = dimK//2
plt.figure()
# plt.grid()
plt.title('average over time, k='+str(k))
for m1 in range(dimM):
    for m2 in range(dimM):
        plt.errorbar(range(L),
                     np.abs(c[:,m1,m2,k]),
                     yerr=c_std[:,m1,m1,k],
                     fmt='o-',
                     markersize=5,
                     capsize=3,
                     label=str(m1)+str(m2),
                     # cmap=cmap
                     c=cmap.colors[m1*dimM+m2]
                     )
plt.legend(loc='right')


plt.figure()
# plt.grid()
k = 300
plt.title('average over time, k='+str(k))
for m in range(dimM):
    plt.errorbar(range(L),
                 np.abs(c[:,m,m,k]),
                 yerr=c_std[:,m,m,k],
                 fmt='o-',
                 markersize=5,
                 capsize=3,
                 label=str(m)+str(m),
                 # cmap=cmap
                 c=cmap.colors[m*dimM]
                 )
plt.legend(loc='right')
plt.show()

cmap = plt.get_cmap("tab20c")
plt.style.use('seaborn-whitegrid')
k = dimK//2
plt.figure()
# plt.grid()
plt.title('average')
for m1 in range(dimM):
    for m2 in range(dimM):
        plt.errorbar(range(L),
                     # np.abs(np.mean(C[:,m1,m2,:,:], axis=(1,2))),
                     np.abs(np.mean(C, axis=(1,2,3,4))),
                     # yerr=c_std[:,m1,m1,k],
                     # fmt='o-',
                     # markersize=5,
                     # capsize=3,
                     label=str(m1)+str(m2),
                     # cmap=cmap
                     c=cmap.colors[m1*dimM+m2]
                     )
plt.legend(loc='right')
plt.show()
