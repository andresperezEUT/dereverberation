"""
This is the code from Archontis to exemplify the System Identification stuff.
The algorithm can be taken directly from here.
STFT-related methods are called on matlab directly (hence the low speed)
"""



# %%
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import matlab.engine

from ctf.ctf_methods import fftconv, stft, istft, fftpartconv, ctfconv


plt.style.use('seaborn-whitegrid')



# %% INIT

L = 100
nSig = round(9.8*L)
nIR = 2*L
nConvSig = nSig+nIR

sig = np.random.randn(nSig)
ir = np.zeros(nIR)
ir[::10] = 1
ir = ir * np.exp(-5*(np.arange(nIR)/nIR))

# plain linear convolution (through full convolution-length FFTs)
convsig = fftconv(ir, sig)
convsig2 = scipy.signal.fftconvolve(ir, sig)
assert np.allclose(convsig, convsig2)



# %%
winsize = 2*L
sigspec = stft(sig, winsize) # spectrum of input
irspec = stft(ir,  winsize) # spectrum of filter
sigconvspec = stft(convsig,  winsize) # spectrum of convolved signal
fooconvsig = istft(sigconvspec)

plt.figure()
plt.subplot(311)
plt.title('signal')
plt.plot(sig)

plt.subplot(3,1,2)
plt.title('filter impulse response')
plt.plot(ir)

plt.subplot(3,1,3)
plt.title('convolved signal')
plt.plot(convsig)
plt.plot(fooconvsig, '-')
plt.show()


# %% comparison betwene partitioned convolution and CTF cirect multiplication

partconvsig = fftpartconv(ir, sig, winsize) # partitioned convolution
ctfconvsig = ctfconv(sigspec, irspec) # convolution through ctf multiplication with fftsize = 2*winsize and Hamming window

plt.figure()
plt.subplot(311)
plt.title('plain FFT convolution')
plt.plot(convsig)

plt.subplot(312)
plt.plot(partconvsig)
plt.title('partitioned convolution')

plt.subplot(313)
plt.plot(ctfconvsig)
plt.title('convolution through CTF multiplication')
plt.show()


# %% system identification (based on windowed STFT)
hopsize = winsize // 2
nBins = winsize + 1
nFrames = sigspec.shape[1]
nFiltFrames = int(np.ceil(nIR/hopsize)) + 1
H = np.zeros((nBins, nFiltFrames), dtype=complex)
sigspec_zpad = np.concatenate( (sigspec, np.zeros((nBins,nFiltFrames-1)) ), axis=1)

for nb in range(nBins):
    y_nb = sigconvspec[nb, nFiltFrames-2+ np.arange(nFrames)]
    X_nb = np.zeros((nFrames,nFiltFrames), dtype=complex)
    for nt in range(nFrames):
        X_nb[nt,:] = np.flip( sigspec_zpad[nb, nt:nt+nFiltFrames] )
    H[nb,:] = np.linalg.pinv(X_nb)@y_nb

ir_est = istft(H,winsize)

plt.figure()
plt.plot(ir)
plt.plot(ir_est,'-')
plt.title('SID with windowed STFT 50% overlap')
plt.show()


# %% non-overlapped STFT (winsize<IR length)
winsize = L
sigspec = stft(sig, winsize, 2*winsize, winsize) # spectrum of input
irspec  = stft(ir,  winsize, 2*winsize, winsize) # spectrum of filter
sigconvspec = stft(convsig,  winsize, 2*winsize, winsize)

fooconvsig = istft(sigconvspec, winsize, winsize)
plt.figure()
plt.plot(convsig)
plt.plot(fooconvsig,'-')
plt.title('convolved signal') # comparison just for STFT/iSTFT validation
plt.show()


# %% system identification (based on non-overlapped STFT)

nFFT = 2*winsize
hopsize = winsize
nBins = winsize+1
nFrames = sigspec.shape[1]
nFiltFrames = int(np.ceil(nIR/hopsize))+1
H = np.zeros((nBins, nFiltFrames), dtype=complex)
sigspec_zpad = np.concatenate( (sigspec, np.zeros((nBins,nFiltFrames-1)) ), axis=1)

for nb in range(nBins):
    y_nb = sigconvspec[nb, nFiltFrames-2+ np.arange(nFrames)]
    X_nb = np.zeros((nFrames,nFiltFrames), dtype=complex)
    for nt in range(nFrames):
        X_nb[nt,:] = np.flip( sigspec_zpad[nb, nt:nt+nFiltFrames] )
    H[nb,:] = np.linalg.pinv(X_nb)@y_nb

ir_est = istft(H,winsize,winsize)
plt.figure()
plt.plot(ir)
plt.plot(ir_est[hopsize:],'-')
plt.title('SID with non-overlapped STFT')
plt.show()


# %% non-overlapped STFT (winsize>=IR length)
winsize = 2*L
sigspec = stft(sig, winsize, 2*winsize, winsize) # spectrum of input
irspec  = stft(ir,  winsize, 2*winsize, winsize) # spectrum of filter
sigconvspec = stft(convsig,  winsize, 2*winsize, winsize)

fooconvsig = istft(sigconvspec, winsize, winsize)
plt.figure()
plt.plot(convsig)
plt.plot(fooconvsig,'--')
plt.title('convolved signal') # comparison just for STFT/iSTFT validation
plt.show()
