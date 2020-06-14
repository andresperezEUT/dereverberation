import numpy as np
from methods import *
import os
import matplotlib.pyplot as plt
import librosa
import scipy.signal
import copy
import librosa.display


#### parameters
L = 1
M = np.power(L+1, 2)

ir_start_time = 0.01
rt60 = 0.03
ir_length = 0.5
fs = 48000

audio_file_length = 2. # seconds
window_size = 256 # samples
window_overlap = window_size//2 # samples
D = int(np.floor(ir_start_time * fs / window_overlap)) # ir start frame
print('D=',D)
if D < 1:
    raise Warning('D should be at least 1!!!')

# std_v = 1e-6
std_v = 0


#### get audio
audio_files = []
data_folder_path = '/Volumes/Dinge/DSD100subset/Sources'
for root, dir, files in os.walk(data_folder_path):
    for f in files:
        extension = os.path.splitext(f)[1]
        if 'wav' in extension:
            audio_files.append(os.path.join(root, f))

#### get ir
ir = generate_late_reverberation_ir_from_rt60(ir_start_time, rt60, ir_length, fs)

drr = direct_reverberant_ratio(ir, ir_start_time, fs)
alpha = decay_rate(rt60, fs)

# late reverb part of the ir
late_ir = copy.copy(ir)
late_ir[0] = 0

plt.title('rt60='+str(rt60) + '  drr='+str(drr))
plt.plot(np.arange(ir_length*fs)/fs,ir)
plt.show()


# analysis
for af in [audio_files[1]]: # todo
    """
    same nomenclature as in Braun
    s: time-domain direct signal
    r: time-domain late reverb signal
    x = s + r : reverberant signal
    v : noise signal
    y = x + v = s + r + v : recorded signal
    """
    audio_file_length_samples = int(audio_file_length * fs)

    # Generate reverberant signals
    s = librosa.core.load(af, sr=fs, mono=True)[0][:audio_file_length_samples]
    r = scipy.signal.fftconvolve(s, late_ir)[:audio_file_length_samples]  # keep original length
    x = s + r
    v = np.zeros(np.size(s)) # todo
    v = np.random.normal(0, std_v, audio_file_length_samples)
    y = x + v

    # STFT of everything
    f, t, S = scipy.signal.stft(s, fs, nperseg=window_size, noverlap=window_overlap)
    _, _, R = scipy.signal.stft(r, fs, nperseg=window_size, noverlap=window_overlap)
    _, _, X = scipy.signal.stft(x, fs, nperseg=window_size, noverlap=window_overlap)
    _, _, V = scipy.signal.stft(v, fs, nperseg=window_size, noverlap=window_overlap)
    _, _, Y = scipy.signal.stft(y, fs, nperseg=window_size, noverlap=window_overlap)

    # True PSD of everything
    # TODO: assert
    PSD_S = np.power(np.abs(S), 2)
    PSD_R = np.power(np.abs(R), 2)
    PSD_X = np.power(np.abs(X), 2)
    PSD_V = np.power(np.abs(V), 2)
    PSD_Y = np.power(np.abs(Y), 2)

    plt.figure()
    plt.title('S')
    librosa.display.specshow(librosa.amplitude_to_db(S,ref=np.max), y_axis='linear', x_axis='time', sr=fs, hop_length=window_overlap)
    plt.colorbar(format='%+2.0f dB')
    plt.show()

    ########

    # IDEAL RATIO MASK
    # from Archontis' notes. todo: find reference
    IRM = np.abs(S)/( np.abs(S) + np.abs(R) )
    IRM_S = IRM * Y
    IRM_R = Y - IRM_S
    _, IRM_s = scipy.signal.istft(IRM_S, fs, nperseg=window_size, noverlap=window_overlap)
    plt.figure()
    plt.title('IRM')
    plt.pcolormesh(t, f, IRM, cmap=librosa.display.cmap(IRM))
    plt.colorbar()
    plt.show()
    # play(IRM_s, fs)
    plt.figure()
    plt.title('IRM_S')
    librosa.display.specshow(librosa.amplitude_to_db(IRM_S,ref=np.max), y_axis='linear', x_axis='time', sr=fs, hop_length=window_overlap)
    plt.colorbar(format='%+2.0f dB')
    plt.show()

    # Oracle Wiener Filter
    oracle_SIR = PSD_S / (PSD_R + PSD_V)
    oracle_WF = oracle_SIR / (oracle_SIR + 1)
    oracle_WF_S = oracle_WF * Y
    oracle_WF_R = Y - oracle_WF_S
    _, oracle_WF_s = scipy.signal.istft(oracle_WF_S, fs, nperseg=window_size, noverlap=window_overlap)
    plt.figure()
    plt.title('oracle_WF')
    plt.pcolormesh(t, f, oracle_WF, cmap=librosa.display.cmap(oracle_WF))
    plt.colorbar()
    plt.show()
    # play(oracle_WF_s, fs)
    plt.figure()
    plt.title('oracle_WF_S')
    librosa.display.specshow(librosa.amplitude_to_db(oracle_WF_S,ref=np.max), y_axis='linear', x_axis='time', sr=fs, hop_length=window_overlap)
    plt.colorbar(format='%+2.0f dB')
    plt.show()

    # DD Wiener Filter with Oracle PSD_R
    beta = 0.98
    oracle_DD_WF, oracle_DD_S = dd_wiener_filter(Y, PSD_R, PSD_V, beta)
    oracle_DD_R = Y - oracle_DD_S
    _, oracle_DD_s = scipy.signal.istft(oracle_DD_S, fs, nperseg=window_size, noverlap=window_overlap)
    plt.figure()
    plt.title('oracle_DD_WF')
    plt.pcolormesh(t, f, oracle_DD_WF, cmap=librosa.display.cmap(oracle_DD_WF))
    plt.colorbar()
    plt.show()
    # play(oracle_DD_s, fs)
    plt.figure()
    plt.title('oracle_DD_S')
    librosa.display.specshow(librosa.amplitude_to_db(oracle_DD_S,ref=np.max), y_axis='linear', x_axis='time', sr=fs, hop_length=window_overlap)
    plt.colorbar(format='%+2.0f dB')
    plt.show()


    ## DD Wiener Filter with Forward Exponential PSD_R estimation
    fw_est_PSD_R = forward_exponential_habets(alpha, window_overlap, D, drr, PSD_Y, PSD_V)
    beta = 0.98
    fw_est_WF, fw_est_S = dd_wiener_filter(Y, fw_est_PSD_R, PSD_V, beta)
    fw_est_R = Y - fw_est_S
    _, fw_est_s = scipy.signal.istft(fw_est_S, fs, nperseg=window_size, noverlap=window_overlap)
    plt.figure()
    plt.title('fw_est_WF')
    plt.pcolormesh(t, f, fw_est_WF, cmap=librosa.display.cmap(fw_est_WF))
    plt.colorbar()
    plt.show()
    # play(fw_est_s, fs)
    plt.figure()
    plt.title('fw_est_S')
    librosa.display.specshow(librosa.amplitude_to_db(fw_est_S,ref=np.max), y_axis='linear', x_axis='time', sr=fs, hop_length=window_overlap)
    plt.colorbar(format='%+2.0f dB')
    plt.show()

    ## COMPARE REVERBERANT SIGNALS
    # plt.figure()
    # plt.title('PSD_R')
    # librosa.display.specshow(librosa.amplitude_to_db(PSD_R,ref=np.max), y_axis='linear', x_axis='time', sr=fs, hop_length=window_overlap)
    # plt.colorbar()
    # plt.show()
    #
    # plt.figure()
    # plt.title('fw_est_PSD_R')
    # librosa.display.specshow(librosa.amplitude_to_db(fw_est_PSD_R,ref=np.max), y_axis='linear', x_axis='time', sr=fs, hop_length=window_overlap)
    # plt.colorbar()
    # plt.show()



    ## EVALUATION
    lsdi = []
    lsdi.append(log_spectral_difference_improvement(Y, S, IRM_S))
    lsdi.append(log_spectral_difference_improvement(Y, S, oracle_WF_S))
    lsdi.append(log_spectral_difference_improvement(Y, S, oracle_DD_S))
    lsdi.append(log_spectral_difference_improvement(Y, S, fw_est_S))
    print(lsdi)

    IRM_r = y - IRM_s
    oracle_WF_r = y - oracle_WF_s
    oracle_DD_r = y - oracle_DD_s
    fw_est_r = y - fw_est_s

    # def segmental_sir(s, r, v):
    #     sir_array = []
    #     hop = 0.1
    #     hop_samples = int(hop*fs)
    #     for idx in range(audio_file_length_samples//hop_samples-1):
    #         start_frame = idx*hop_samples
    #         end_frame = (idx+1)*hop_samples
    #         sir = np.abs(s[start_frame:end_frame]) / np.abs((r[start_frame:end_frame]) + np.abs(v[start_frame:end_frame]))
    #         sir_array.append(np.mean(sir))
    #     return 10*np.log10(np.asarray(sir_array))
    #     # return (np.asarray(sir_array))

    # plt.figure()
    # plt.plot(segmental_sir(s, r, v)[1:], label='true')
    # plt.plot(segmental_sir(IRM_s, IRM_r, v)[1:], label='IRM')
    # plt.plot(segmental_sir(oracle_WF_s, oracle_WF_r, v)[1:], label='oracle_WF')
    # plt.plot(segmental_sir(oracle_DD_s, oracle_DD_r, v)[1:], label='oracle_DD')
    # plt.plot(segmental_sir(fw_est_s, fw_est_r, v)[1:], label='fw_est')
    # plt.legend()
    # plt.grid()
    # plt.show()

    # h = 0.02
    # plt.figure()
    # plt.title('segmental SRR')
    # plt.plot(segmental_SRR(s, IRM_s, fs, h), label='IRM')
    # plt.plot(segmental_SRR(s, oracle_WF_s, fs, h), label='oracle_WF')
    # plt.plot(segmental_SRR(s, oracle_DD_s, fs, h), label='oracle_DD')
    # plt.plot(segmental_SRR(s, fw_est_s, fs, h), label='fw_est')
    # plt.legend()
    # plt.grid()
    # plt.show()


    # h = 0.02
    # plt.figure()
    # plt.title('segmental SRR')
    # plt.plot(segmental_SRR(x, s, fs, h), label='true')
    # plt.plot(segmental_SRR(x, IRM_s, fs, h), label='IRM')
    # plt.plot(segmental_SRR(x, oracle_WF_s, fs, h), label='oracle_WF')
    # plt.plot(segmental_SRR(x, oracle_DD_s, fs, h), label='oracle_DD')
    # plt.plot(segmental_SRR(x, fw_est_s, fs, h), label='fw_est')
    # plt.legend()
    # plt.grid()
    # plt.show()
    #
    # play(fw_est_s, fs)

    ## PSD_R ESTIMATION ERROR
    e, m, lv, uv = log_PSD_error(PSD_R, fw_est_PSD_R, D)

    plt.figure()
    plt.title('PSD_R')
    librosa.display.specshow(librosa.power_to_db(PSD_R,ref=np.max), y_axis='linear', x_axis='time', sr=fs, hop_length=window_overlap)
    plt.colorbar(format='%+2.0f dB')
    plt.show()

    plt.figure()
    plt.title('fw_est_PSD_R')
    librosa.display.specshow(librosa.power_to_db(fw_est_PSD_R,ref=np.max), y_axis='linear', x_axis='time', sr=fs, hop_length=window_overlap)
    plt.colorbar(format='%+2.0f dB')
    plt.show()

    plt.figure()
    plt.title('PSD_R estimation error')
    plt.pcolormesh(e, cmap=librosa.display.cmap(e))
    plt.colorbar()
    plt.show()


    ## SIR
    true_input_SIR = PSD_S[:,D:] / PSD_R[:,D:]
    IRM_SIR = np.power(np.abs(IRM_S[:,D:]), 2) / np.power(np.abs(IRM_R[:,D:]), 2)
    oracle_WF_SIR = np.power(np.abs(oracle_WF_S[:,D:]), 2) / np.power(np.abs(oracle_WF_R[:,D:]), 2)
    oracle_DD_SIR = np.power(np.abs(oracle_DD_S[:,D:]), 2) / np.power(np.abs(oracle_DD_R[:,D:]), 2)
    fw_est_SIR = np.power(np.abs(fw_est_S[:,D:]), 2) / np.power(np.abs(fw_est_R[:,D:]), 2)

    np.mean(10*np.log10(true_input_SIR))
    np.mean(10*np.log10(IRM_SIR))
    np.mean(10*np.log10(oracle_WF_SIR))
    np.mean(10*np.log10(oracle_DD_SIR))
    np.mean(10*np.log10(fw_est_SIR))


    import mir_eval.separation

    d = int(np.floor(ir_start_time * fs))
    input_SDR = mir_eval.separation.bss_eval_sources(s[d:], y[d:])[0][0]
    IRM_SDR = mir_eval.separation.bss_eval_sources(s[d:], IRM_s[d:])[0][0]
    oracle_WF_SDR = mir_eval.separation.bss_eval_sources(s[d:], oracle_WF_s[d:])[0][0]
    oracle_DD_SDR = mir_eval.separation.bss_eval_sources(s[d:], oracle_DD_s[d:])[0][0]
    fw_est_SDR = mir_eval.separation.bss_eval_sources(s[d:], fw_est_s[d:])[0][0]

    print('SDRi (the bigger the better)')
    print('  IRM,        oracle_WF,  oracle_DD, fw_est')
    print([IRM_SDR, oracle_WF_SDR, oracle_DD_SDR, fw_est_SDR] - input_SDR)

    input_LSD = log_spectral_difference(S[:,D:], Y[:,D:])
    IRM_LSD = log_spectral_difference(S[:,D:], IRM_S[:,D:])
    oracle_WF_LSD = log_spectral_difference(S[:,D:], oracle_WF_S[:,D:])
    oracle_DD_LSD = log_spectral_difference(S[:,D:], oracle_DD_S[:,D:])
    fw_est_LSD = log_spectral_difference(S[:,D:], fw_est_S[:,D:])

    print('LSD (the lower the better)')
    print('  IRM,        oracle_WF,  oracle_DD, fw_est')
    print([IRM_LSD, oracle_WF_LSD, oracle_DD_LSD, fw_est_LSD] - input_LSD)
