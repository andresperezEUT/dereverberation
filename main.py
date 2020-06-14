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

pp = "/Users/andres.perez/source/masp"
sys.path.append(pp)
import masp


# Get audio files
#
# fs = 48000
# start_time = 0.1
# rt60 = 1.5
# length = 2
# fs = 16000
# ir = generate_late_reverberation_ir_from_rt60(start_time, rt60, length, fs)
#
# plt.figure()
# plt.plot(ir)
# plt.show()
#
#
audio_files = []
data_folder_path = '/Volumes/Dinge/DSD100subset/Sources'
for root, dir, files in os.walk(data_folder_path):
    for f in files:
        extension = os.path.splitext(f)[1]
        if 'wav' in extension:
            audio_files.append(os.path.join(root, f))
#
# # Convolve with irs
# audio_file_length = 2. ## seconds
# audio_file_length_samples = int(audio_file_length*fs)
# af = audio_files[0]
# s_dir = librosa.core.load(af, sr=fs, mono=True)[0][:audio_file_length_samples]
#
# wet_sig = scipy.signal.fftconvolve(s_dir, ir)[:audio_file_length_samples]  # keep original length
#
# plt.figure()
# plt.subplot(2,2,1)
# plt.plot(np.arange(audio_file_length_samples)/fs, s_dir)
# # plt.plot(s_dir)
# plt.grid()
# plt.title('Dry signal')
#
# plt.subplot(2,2,3)
# D = np.abs(librosa.stft(s_dir))
# librosa.display.specshow(librosa.amplitude_to_db(D,ref=np.max), y_axis='log', x_axis='time', sr=fs)
# plt.title('Dry signal')
# plt.colorbar(format='%+2.0f dB')
# plt.tight_layout()
#
# plt.subplot(2,2,2)
# plt.plot(np.arange(audio_file_length_samples)/fs, wet_sig)
# # plt.plot(wet_sig)
# plt.grid()
# plt.title('Wet signal')
#
# plt.subplot(2,2,4)
# D = np.abs(librosa.stft(wet_sig))
# librosa.display.specshow(librosa.amplitude_to_db(D,ref=np.max), y_axis='log', x_axis='time', sr=fs)
# plt.title('Wet signal')
# plt.colorbar(format='%+2.0f dB')
# plt.tight_layout()
#
# plt.show()


####################################

## GET RANDOM AMBISONIC DIRECTION

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

start_time = 0.02
rt60 = 0.5
length = 1
fs = 48000
irs = np.zeros((M, length*fs))

# rt60_vector = np.arange(0.1, 1.6, 0.1)
rt60_vector = [0.5]
for af in [audio_files[1]]: # todo
    lsdi_vector = []
    for rt60 in rt60_vector:

        print('rt60', rt60)

        for m in range(M):
            irs[m] = generate_late_reverberation_ir(start_time, rt60, length, fs)
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
        directivity = S_tot_ambi.compute_ita_re(r=r)

        psa.plot_signal(s_tot_ambi)
        psa.plot_magnitude_spectrogram(S_tot_ambi)
        psa.plot_doa(doa)
        psa.plot_directivity(directivity)
        # psa.plot_directivity(directivity.sqrt())

        est_S_dir_ambi = S_tot_ambi.apply_mask(directivity.sqrt())
        est_S_dir = est_S_dir_ambi.data[0]
        # psa.plot_magnitude_spectrogram(est_S_dir_ambi)
        doa_est_S_dir_ambi = psa.compute_DOA(est_S_dir_ambi)
        directivity_est_S_dir_ambi = est_S_dir_ambi.compute_ita_re(r=r)
        psa.plot_doa(doa_est_S_dir_ambi,title='after method')
        # psa.plot_directivity(directivity_est_S_dir_ambi)
        plt.show()

        est_s_dir_ambi = psa.Signal.fromStft(est_S_dir_ambi,
                                       window_size=window_size,
                                       window_overlap=window_overlap
                                       )

        est_s_dir = est_s_dir_ambi.data[0]

        mask = np.sqrt(directivity.data[0])
        inv_mask = 1 - mask
        est_S_rev = S_tot_ambi.data[0] * inv_mask

        # est_S_rev = S_tot_ambi.data[0] - est_S_dir_ambi.data[0]
        # est_S_rev = S_tot_ambi.data[0] - est_S_dir_ambi.data[0]



        _, est_s_rev = scipy.signal.istft(est_S_rev, fs, nperseg=window_size, noverlap=window_overlap)


        S_rev = S_tot_ambi.data[0] - S_dir
        _, s_rev = scipy.signal.istft(S_rev, fs, nperseg=window_size, noverlap=window_overlap)

        # psa.plot_signal(s_tot_ambi)
        # psa.plot_signal(est_s_dir_ambi)
        # plt.show()


        # # ## original
        # play(s_dir)
        # # ## wet W=
        # play(s_tot_ambi.data[0])
        # # reverb
        # play(s_rev)
        # # ## dereverb W
        # play(est_s_dir_ambi.data[0])


        ### METRICS
        def LSD(a, b):
            ## a,b are matrices of same shape
            return np.sqrt(np.mean(np.power(20*np.log10(np.abs(a/b)), 2)))

        def LSDI(reverberant, reference, estimated):
            return LSD(reverberant, reference) - LSD(estimated, reference)


        # lsdi = LSDI(S_tot_ambi.data[0], S_dir, est_S_dir_ambi.data[0])
        # print('LSDI', lsdi)
        # lsdi_vector.append(lsdi)
        # print('------')

        ## DRR
        # DRR = direct/reverberant

        DRR = np.mean(20*np.log(np.abs(S_dir/S_rev)))
        DRR_estimated = np.mean(20*np.log(np.abs(est_S_dir_ambi.data[0]/S_rev)))


    # plt.figure()
    # plt.title(af+'ldsi versus rt60')
    # plt.plot(rt60_vector,lsdi_vector,'*-')
    # plt.grid()
    # plt.show()

IRM = np.abs(S_dir)/( np.abs(S_dir) + np.abs(S_rev) )


f,t = S_dir.shape
IRM_inverse = 1-IRM

oracle_S_dir = IRM*S_tot_ambi.data[0]
oracle_S_rev = IRM_inverse*S_tot_ambi.data[0]

_, oracle_s_dir = scipy.signal.istft(oracle_S_dir, fs, nperseg=window_size, noverlap=window_overlap)
_, oracle_s_rev = scipy.signal.istft(oracle_S_rev, fs, nperseg=window_size, noverlap=window_overlap)



## DDR

# DRR = 10*np.log( np.mean(np.power(np.abs(S_dir),2)) / np.mean(np.power(np.abs(S_rev),2)) )
# oracle_DDR = 10*np.log( np.mean(np.power(np.abs(oracle_S_dir),2)) / np.mean(np.power(np.abs(oracle_S_rev),2)) )
# est_DRR = 10*np.log( np.mean(np.power(np.abs(est_S_dir),2)) / np.mean(np.power(np.abs(est_S_rev),2)) )
# print(DRR,oracle_DDR,est_DRR )
# same results in time-domain...
DRR = 10*np.log( np.mean(np.power(np.abs(s_dir),2)) / np.mean(np.power(np.abs(s_rev),2)) )
oracle_DDR = 10*np.log( np.mean(np.power(np.abs(oracle_s_dir),2)) / np.mean(np.power(np.abs(oracle_s_rev),2)) )
est_DRR = 10*np.log( np.mean(np.power(np.abs(est_s_dir),2)) / np.mean(np.power(np.abs(est_s_rev),2)) )
print(DRR,oracle_DDR,est_DRR )
# #
# ## input
# play(s_dir)
# play(s_rev)
# play(s_tot_ambi.data[0])
#
# # IRM
# play(oracle_s_dir)
# play(oracle_s_rev)
#
# # ESTIMATED
# play(est_s_dir)
# play(est_s_rev)


################################
############ PLOTS

plt.figure()
plt.title('S_dir')
plt.pcolormesh(20*np.log10(np.abs(S_dir)), vmin=-100, vmax=0)
plt.colorbar()
plt.show()

plt.figure()
plt.title('S_rev')
plt.pcolormesh(20*np.log10(np.abs(S_rev)), vmin=-100, vmax=0)
plt.colorbar()
plt.show()

plt.figure()
plt.title('IRM')
plt.pcolormesh(IRM)
plt.colorbar()
plt.show()

plt.figure()
plt.title('oracle_S_dir')
plt.pcolormesh(20*np.log10(np.abs(oracle_S_dir)), vmin=-100, vmax=0)
plt.colorbar()
plt.show()

plt.figure()
plt.title('oracle_S_rev')
plt.pcolormesh(20*np.log10(np.abs(oracle_S_rev)), vmin=-100, vmax=0)
plt.colorbar()
plt.show()

plt.figure()
plt.title('est_S_dir')
plt.pcolormesh(20*np.log10(np.abs(est_S_dir)), vmin=-100, vmax=0)
plt.colorbar()
plt.show()

plt.figure()
plt.title('est_S_rev')
plt.pcolormesh(20*np.log10(np.abs(est_S_rev)), vmin=-100, vmax=0)
plt.colorbar()
plt.show()


##########


def single_channel_oracle_wiener_filter(S_dir, S_rev, S_tot):

    # TODO: IS THAT CORRECT??????
    PSD_noise = 0
    PSD_dir = np.power(np.abs(S_dir), 2)
    PSD_rev = np.power(np.abs(S_rev), 2)

    WF = PSD_dir / ( PSD_dir + PSD_rev + PSD_noise)
    est_S_dir = S_tot * WF

    return est_S_dir

oracle_WF_S_dir = single_channel_oracle_wiener_filter(S_dir, S_rev, S_tot_ambi.data[0])
plt.pcolormesh(20*np.log10(np.abs(oracle_WF_S_dir)), vmin=-100, vmax=0)
plt.colorbar()
plt.show()


_, oracle_WF_s_dir = scipy.signal.istft(oracle_WF_S_dir, fs, nperseg=window_size, noverlap=window_overlap)
play(oracle_WF_s_dir)

#########

def a_posteriori_SIR(Y, psd_RD, psd_V):
    """
    Y: recorded
    psd_RD (psd of the late reverb)
    psd_V (psd of the noise)
    gamma_D(k,n) (3.7)
    :return:
    """
    return (np.power(np.abs(Y), 2)) / (psd_RD + psd_V)

def a_priori_SIR(est_SD, psd_RD, psd_V):
    """
    est_SD: estimated SD
    psd_RD (psd of the late reverb)
    psd_V (psd of the noise)
    ji_tilde_D (3.7)
    :return:
    """
    return (np.power(np.abs(est_SD), 2)) / (psd_RD + psd_V)


def decission_directed_a_priori_SIR(est_SD_last, psd_RD_last, psd_V_last, Y, psd_RD, psd_V, beta):
    """
    est_ji_D
    :param Y:
    :param psd_RD:
    :param psd_V:
    :param beta:
    :return:
    """

    ji_tilde_D = a_priori_SIR(est_SD_last, psd_RD_last, psd_V_last)
    gamma_D = a_posteriori_SIR(Y, psd_RD, psd_V)
    return beta * ji_tilde_D + (1-beta) * np.max(gamma_D-1, 0)


def wiener_filter_dd(est_SD_last, psd_RD_last, psd_V_last, Y, psd_RD, psd_V, beta):

    est_ji_D = decission_directed_a_priori_SIR(est_SD_last, psd_RD_last, psd_V_last, Y, psd_RD, psd_V, beta)
    return est_ji_D / (est_ji_D + 1)



Y = S_tot_ambi.data[0] # recorded data
K, N = Y.shape
psd_RD = np.power(np.abs(S_rev), 2) # todo: we will estimate this one
psd_V = np.zeros((K,N))
psd_V_last = 0

beta = 0.9

est_SD_wiener_last = np.zeros(K)
psd_RD_last = np.zeros(K)

WF = np.zeros((K,N))
est_SD_wiener = np.zeros((K,N))

for n in range(1, N):
    # WF[:,n] = wiener_filter_dd(est_SD_wiener_last, psd_RD_last, psd_V_last, Y[:,n], psd_RD[:,n], psd_V, beta)
    WF[:,n] = wiener_filter_dd(est_SD_wiener[:,n-1], psd_RD[:,n-1], psd_V[:,n-1], Y[:,n], psd_RD[:,n], psd_V[:,n], beta)
    est_SD_wiener[:,n] = WF[:,n] * Y[:,n]  # Eq 3.4

plt.pcolormesh(20*np.log10(np.abs(est_SD_wiener)), vmin=-100, vmax=0)
plt.colorbar()
plt.show()

_, est_sD_wiener = scipy.signal.istft(est_SD_wiener, fs, nperseg=window_size, noverlap=window_overlap)
play(est_sD_wiener)



