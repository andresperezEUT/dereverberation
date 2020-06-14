# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2019, Eurecat / UPF
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the <organization> nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#   @file   test_script_mics.py
#   @author Andrés Pérez-López
#   @date   29/07/2019
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
import sys



pp = "/Users/andres.perez/source/masp"
sys.path.append(pp)
import masp
from masp import shoebox_room_sim as srs
import time
import librosa
import matplotlib.pyplot as plt
import math
import statistics


# %%
from test_dif_2 import dist
# import dist
dist(1,2,3,4)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# SETUP

# Room definition

width = 6
height = 3
depth = 10

room = np.array([depth, width, height])

# Desired RT per octave band, and time to truncate the responses
rt60 = np.array([1., 0.8, 0.7, 0.6, 0.5, 0.4])
nBands = len(rt60)

# Generate octave bands
band_centerfreqs = np.empty(nBands)
band_centerfreqs[0] = 125
for nb in range(1, nBands):
    band_centerfreqs[nb] = 2 * band_centerfreqs[nb - 1]

# Absorption for approximately achieving the RT60 above - row per band
abs_wall = srs.find_abs_coeffs_from_rt(room, rt60)[0]

# Critical distance for the room
_, d_critical, _ = srs.room_stats(room, abs_wall)

# Receiver position
rec = np.array([[9.0, 3.0, 1.5]])
nRec = rec.shape[0]

# Source positions
src = np.array([[1.0, 2.0, 1.5], [4.0, 3.0, 1.5], [7.0, 4.0, 1.5]])
nSrc = src.shape[0]

# Mic orientations and directivities
mic_specs = np.array([[1, 0, 0, 1]])  # Omnidirectional Microphone

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# RUN SIMULATOR

# Echogram
tic = time.time()

maxlim = 1.5  # just stop if the echogram goes beyond that time ( or just set it to max(rt60) )
limits = np.minimum(rt60, maxlim)

# Compute echograms
# abs_echograms, rec_echograms, echograms = srs.compute_echograms_mic(room, src, rec, abs_wall, limits, mic_specs);
abs_echograms = srs.compute_echograms_mic(room, src, rec, abs_wall, limits, mic_specs);

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# RENDERING

# In this case all the information (receiver directivity especially) is already
# encoded in the echograms, hence they are rendered directly to discrete RIRs
fs = 48000
mic_rirs = srs.render_rirs_mic(abs_echograms, band_centerfreqs, fs)

toc = time.time()
print('Elapsed time is ' + str(toc - tic) + 'seconds.')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# GENERATE SOUND SCENES
# Each source is convolved with the respective mic IR, and summed with
# the rest of the sources to create the microphone mixed signals

sourcepath = '/Users/andres.perez/source/MATLAB/polarch/shoebox-roomsim-master/milk_cow_blues_4src.wav'
src_sigs = librosa.core.load(sourcepath, sr=None, mono=False)[0].T[:, :nSrc]

mic_sigs = srs.apply_source_signals_mic(mic_rirs, src_sigs)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# COMPUTE THE ACOUSTIC PARAMETERS

# plt.plot(mic_rirs[:,0,0]) #Impulse Response

print("-------- PARAMETRES ACUSTICS --------")

# RT60
Temperature = 20
humidity = 25
c = 331.4 + 0.6 * Temperature
V = room[0] * room[1] * room[2]  # Volume of the class
Q = 2  # Directivity Factor for speech in a class

# Assuming there are no windows and door. In that case substract the S from it
# and sum the S to the total S.
s_Ceiling = width * depth
s_Floor = width * depth
s_FrontWall = width * height
s_BackWall = width * height
s_RightWall = depth * height
s_LeftWall = depth * height
superficie = [s_Ceiling, s_Floor, s_FrontWall, s_BackWall, s_RightWall, s_LeftWall]
S = s_Ceiling + s_Floor + s_FrontWall + s_BackWall + s_RightWall + s_LeftWall

coefAbs = np.matrix(
    [[0.57, 0.39, 0.41, 0.82, 0.89, 0.72], [0.2, 0.15, 0.12, 0.1, 0.1, 0.07], [0.01, 0.01, 0.02, 0.03, 0.04, 0.05],
     [0.01, 0.01, 0.02, 0.03, 0.04, 0.05], [0.01, 0.01, 0.02, 0.03, 0.04, 0.05], [0.01, 0.01, 0.02, 0.03, 0.04, 0.05]])
coefAbsMig = []
for a in range(0, nBands):
    coefmigband = np.dot(superficie, coefAbs[:, a]) / S
    coefAbsMig.append(coefmigband)

m = []
for a in range(0, nBands):
    coef_aire = 5.5 * (10 ** -4) * (50 / humidity) * ((band_centerfreqs[a] / 1000) ** 1.7)  # Coeficient de l'Aire
    m.append(coef_aire)

RT60 = []
for e in range(0, nBands):
    if V < 500 and coefAbsMig[e] < 0.2:  # No air coef and Sabine
        rt = (60 * V) / (1.086 * c * S * coefAbsMig[e])
        RT60.append(rt)
    elif V < 500 and coefAbsMig[e] > 0.2:  # No air coef and Eyring
        rt = (60 * V) / (1.086 * c * S * (-math.log(1 - coefAbsMig[e])))
        RT60.append(rt)
    elif V > 500 and coefAbsMig[e] < 0.2:  # Air coef and Sabine
        rt = (60 * V) / (1.086 * c * (S * coefAbsMig[e] + 4 * m * V))
        RT60.append(rt)
    else:  # Air coef and Eyring
        rt = (60 * V) / (1.086 * c * S * (-math.log(1 - coefAbsMig[e]) + (4 * m * V) / S))
        RT60.append(rt)

R = []
for d in range(0, nBands):
    ct_sala = (S * coefAbsMig[d]) / (1 - coefAbsMig[d])
    R.append(ct_sala)

Dc = []
for k in range(0, nBands):
    dist_critica = math.sqrt((Q * R[k]) / (16 * math.pi))
    Dc.append(dist_critica)
# RT = statistics.mean(rt60)
RT = rt60[3]  # RT (1kHz)

BR = (rt60[0] + rt60[1]) / (rt60[2] + rt60[3])  # Calidesa acustica (BR) Objective: 0.9 - 1
print("Calidesa acustica (BR): " + str('%.3f' % BR))

Br = (rt60[4] + rt60[5]) / (rt60[2] + rt60[3])  # Brillantor (Br) Objective: >0.80
print("Brillantor (Br): " + str('%.3f' % Br))

samples_b50 = mic_rirs[:int(0.05 * fs), 0, 0]
Energy_b50 = sum(map(lambda i: i * i, samples_b50))
samples_a50 = mic_rirs[int(0.05 * fs):, 0, 0]
Energy_a50 = sum(map(lambda i: i * i, samples_a50))
C50 = 10 * math.log10(Energy_b50 / Energy_a50)
print("Index de Claredat (C50): " + str('%.3f' % C50))

D50 = 1 / (1 + 10 ** -(C50 / 10))  # Definicio de la veu (D50) Objective: 0.4 - 0.6
print("Definicio de la veu (D50): " + str('%.3f' % D50))

# Perdua de l’Articulacio de les Consonants (%ALCons) Objective: 0 - 7%
ALCons_r = []
r = []

print("----- ALCONS with different distance between src-rec and fixed RT60 -----")

# Compute %ALCONS with different distance between src-rec and fixed RT60
for x in range(0, len(src)):
    distance = math.sqrt((src[x, 0] - rec[0, 0]) ** 2 + (src[x, 1] - rec[0, 1]) ** 2 + (src[x, 2] - rec[0, 2]) ** 2)
    r.append(distance)

for y in range(0, len(r)):
    if r[y] <= 3.16 * Dc[3]:
        Cons_r = (200 * (r[y] ** 2) * (RT ** 2)) / (V * Q)
        ALCons_r.append(Cons_r)
    else:
        Cons_r = 9 * rt60[3]
        ALCons_r.append(Cons_r)
    print("%ALCons: " + str('%.3f' % Cons_r) + " a una distància: " + str('%.3f' % r[y]))

# Compute %ALCONS with different rt60 and fixed src-rec distance

print("----- ALCONS with different rt60 and fixed src-rec distance -----")

ALCons_60 = []
for z in range(0, len(rt60)):
    if r[y] <= 3.16 * Dc[3]:
        Cons_60 = (200 * (r[1] ** 2) * (rt60[z] ** 2)) / (V * Q)
        ALCons_60.append(Cons_60)
    else:
        Cons_60 = 9 * rt60[3]
        ALCons_60.append(Cons_60)
    print("%ALCons: " + str('%.3f' % Cons_60) + " amb un RT60 de: " + str(rt60[z]))

print("----- Speech Sound Level with different src-rec distance -----")
Lw = 94
Smid = []
for j in range(0, len(r)):
    Lp = Lw - abs(10 * math.log(Q / 4 * math.pi * (r[j] ** 2)))
    smid = Lp - Lw + 39
    Smid.append(smid)
    print("S: " + str('%.3f' % smid) + " a una distancia: " + str(r[j]))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# PLOT PARAMETERS DEPENDENCIES
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %ALCons vs Source-Receiver
plt.plot(ALCons_r, r, color='r', linestyle='solid', marker='o', linewidth=1, markersize=8)
plt.grid()
plt.xlabel('%ALCons [%]');
plt.ylabel('Distància Source-Receiver [meters]');
plt.title('%ALCons en funció de la distància entre emissor i receptor')
plt.show()
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %ALCons vs RT60
plt.plot(ALCons_60, rt60, color='g', linestyle='solid', marker='o', linewidth=1, markersize=8)
plt.grid()
plt.xlabel('%ALCons [%]');
plt.ylabel('RT60 [seconds]');
plt.title('%ALCons en funció del RT60')
plt.show()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %Smid vs Source-Receiver
plt.plot(Smid, r, color='b', linestyle='solid', marker='o', linewidth=1, markersize=8)
plt.grid()
plt.xlabel('Speech Sound Level');
plt.ylabel('Distància Source-Receiver [meters]');
plt.title('Sonoritat en funció de la distància entre emissor i receptor')
plt.show()
