"""
Make an experiment comparing different techniques for estimating RT60 from an IR.
Plot mean error and deviations for all methods.
Results are saved in /test_rt60/data.
Conclusion: method `compute_t60` with 't10' parameter perform best
"""


# %%
import numpy as np
import sys

from ctf.ctf_methods import rt60_bands
from ctf.ctf_methods import compute_t60_acoustics
from ctf.ctf_methods import compute_t60

pp = "/Users/andres.perez/source/masp"
sys.path.append(pp)
import masp
from masp import shoebox_room_sim as srs

import matplotlib.pyplot as plt



# %%

rt60_decay = 0.05
nBands = 5
band_centerfreqs = np.empty(nBands)
band_centerfreqs[0] = 250
for nb in range(1, nBands):
    band_centerfreqs[nb] = 2 * band_centerfreqs[nb - 1]

azi = np.random.rand()*2*np.pi
incl = np.random.rand()*np.pi
sh_order = 0
fs = 16000
dimM = 1



# %%

I = 21 # num iterations

rt60_types = ['edt', 't10', 't20', 't30']
rt60_true = np.zeros((I, nBands))
results_acoustics = np.zeros((I, nBands, len(rt60_types)))
results_mine = np.zeros((I, nBands, len(rt60_types)))


# %%

for i in range(I):

    # between 0.4 and 1.3
    rt60_0 = np.random.rand() + 0.3

    print('------------------------------')
    print('i = ', i)
    print('rt60 = ', rt60_0)

    room = np.array([10.2, 7.1, 3.2])
    rt60 = rt60_bands(rt60_0, nBands, rt60_decay)
    rt60_true[i, :] = rt60

    abs_wall = srs.find_abs_coeffs_from_rt(room, rt60)[0]

    # Critical distance for the room
    _, d_critical, _ = srs.room_stats(room, abs_wall, verbose=False)

    # Receiver position
    rec = (room / 2)[np.newaxis]  # center of the room
    nRec = rec.shape[0]

    # d_critical distance with defined angular position
    azi = azi + np.pi  # TODO: fix in srs library!!!
    src_sph = np.array([azi, np.pi / 2 - incl, d_critical.mean()])
    src_cart = masp.sph2cart(src_sph)
    src = rec + src_cart
    nSrc = src.shape[0]

    # SH orders for receivers
    rec_orders = np.array([sh_order])

    maxlim = 1.3  # just stop if the echogram goes beyond that time ( or just set it to max(rt60) )
    limits = np.ones(nBands) * maxlim  # hardcoded!

    abs_echograms = srs.compute_echograms_sh(room, src, rec, abs_wall, limits, rec_orders)
    irs = srs.render_rirs_sh(abs_echograms, band_centerfreqs, fs).squeeze().T
    if irs.ndim == 1:
        irs = irs[np.newaxis, :]
    # Normalize as SN3D
    irs *= np.sqrt(4 * np.pi)



    # %%

    ir = irs[0]
    for t_i, t in enumerate(rt60_types):
        results_acoustics[i, :, t_i] = compute_t60_acoustics(ir, fs, band_centerfreqs, rt=t)
        results_mine[i, :, t_i] = compute_t60(ir, fs, band_centerfreqs, rt=t)





# %%

col = plt.rcParams['axes.prop_cycle'].by_key()['color']

for f_idx in range(nBands):
    f = band_centerfreqs[f_idx]

    plt.figure()
    plt.title('all rt60 estimations, f=' + str(f))
    plt.grid()
    plt.plot(rt60_true[:,f_idx])  # true rt6 at 1k for all iterations
    plt.plot(results_acoustics[:,f_idx], linestyle='--')
    plt.plot(results_mine[:,f_idx], linestyle=':')

    plt.figure()
    plt.title('method differences, f=' + str(f))
    plt.grid()
    for t in range(len(rt60_types)):
        plt.plot(results_acoustics[:,f_idx, t] - rt60_true[:,f_idx] , linestyle='--', label=rt60_types[t], c=col[t+1])
        mean = np.mean(results_acoustics[:,f_idx, t] - rt60_true[:,f_idx])
        plt.axhline(mean, 0 , I-1, linestyle='--', c=col[t+1])

    for t in range(len(rt60_types)):
        plt.plot(results_mine[:,f_idx, t] - rt60_true[:,f_idx] , linestyle=':', label=rt60_types[t], c=col[t+5])
        mean = np.mean(results_mine[:,f_idx, t] - rt60_true[:,f_idx])
        plt.axhline(mean, 0 , I-1, linestyle=':', c=col[t+5])

    plt.legend()

    # t10_mine looks like the best option!!

    # np.save('/Users/andres.perez/source/dereverberation/ctf/test_rt60_data/rt60_true.npy',rt60_true)
    # np.save('/Users/andres.perez/source/dereverberation/ctf/test_rt60_data/results_acoustics.npy',results_acoustics)
    # np.save('/Users/andres.perez/source/dereverberation/ctf/test_rt60_data/results_mine.npy',results_mine)