import numpy as np
import sys
pp = "/Users/andres.perez/source/masp"
sys.path.append(pp)
import masp
pp = "/Users/andres.perez/source/parametric_spatial_audio_processing"
sys.path.append(pp)
import parametric_spatial_audio_processing as psa
import matplotlib.pyplot as plt

# Diffuseness experiment

fs = 16000
audio_length = 1.
audio_length_samples = int(audio_length * fs)
num_channels = 2
C = 3

random_dir = np.random.rand(3) - 0.5
azi0, ele0, _ = masp.cart2sph(random_dir)
incl0 = np.pi/2 - ele0
anti_azi0 = azi0 + np.pi
anti_ele0 = -ele0

num_steps = 10
azis = np.linspace(azi0, anti_azi0, num_steps)
eles = np.linspace(ele0, anti_ele0, num_steps)

# azis = np.arange(0,np.pi+0.001,np.pi/20.)
mean_psi_vector = np.empty(num_steps)
mean_ita_vector = np.empty(num_steps)
mean_msc_vector = np.empty((num_steps, C))
mean_m_vector = np.empty((num_steps, C))
mean_msw_vector = np.empty((num_steps, C))


def dist(az1, ele1, az2, ele2):
    '''
    great circle distance
    '''
    dist = np.sin(ele1) * np.sin(ele2) + np.cos(ele1) * np.cos(ele2) * np.cos(np.abs(az1 - az2))
    # Making sure the dist values are in -1 to 1 range, else np.arccos kills the job
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist

for idx in range(num_steps):

    # # # # # # # # #
    azi1 = azis[idx]
    ele1 = eles[idx]
    incl1 = np.pi/2 - ele1
    print(azi1, ele1)

    distance = dist(azi0, ele0, azi1, ele1)
    print('distance: ', distance)
    # # # # # # # # #

    d = np.asarray([[azi0, incl0], [azi1, incl1]])
    basisType = 'real'
    y = masp.get_sh(1, d, basisType) * np.sqrt(4*np.pi) * [1, 1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)] ## ACN, SN3D

    s = np.random.normal(size=(num_channels,audio_length_samples))
    ambi0 = s[0][:,np.newaxis] * y[0]
    ambi1 = s[1][:,np.newaxis] * y[1]

    ambi = ambi0+ambi1


    # # # # # # # # # # # # # # # # # #
    r = 1
    window_size = 256
    window_overlap = window_size // 2

    s_tot_ambi = psa.Signal(ambi.T, fs, 'acn', 'n3d')
    S_tot_ambi = psa.Stft.fromSignal(s_tot_ambi,
                                     window_size=window_size,
                                     window_overlap=window_overlap
                                     )
    doa = psa.compute_DOA(S_tot_ambi)
    # ksi = S_tot_ambi.compute_ksi(r=r)
    msc = S_tot_ambi.compute_msc(r=r)
    msw = S_tot_ambi.compute_msw(r=r)
    ksi = np.dot(msc,msw)
    m = np.asarray([msc.data[0] * msw.data[0], msc.data[1] * msw.data[1], msc.data[2] * msw.data[2]])
    A = np.sqrt(msc.data[0] * msw.data[0] + msc.data[1] * msw.data[1] + msc.data[2] * msw.data[2])
    ksi = psa.Stft(msc.t, msc.f, A, msc.sample_rate)
    ita = S_tot_ambi.compute_ita(r=r)

    # ksi = np.sqrt(msc.data[0] * msw.data[0] + msc.data[1] * msw.data[1] + msc.data[2] * msw.data[2])

    # psa.plot_doa(doa, title=str(idx))
    # psa.plot_directivity(msc, title='MSC '+str(idx))
    # psa.plot_directivity(msw, title='MSW '+str(idx))
    # psa.plot_directivity(ksi, title='KSI '+str(idx))
    # plt.show()

    mean_psi_vector[idx] = 1 - np.mean(ksi.data)
    mean_ita_vector[idx] = np.mean(ita.data)
    mean_m_vector[idx] = np.mean(m, axis=(1,2))
    mean_msc_vector[idx] = np.mean(msc.data, axis=(1,2))
    mean_msw_vector[idx] = np.mean(msw.data, axis=(1,2))


# # # m
plt.figure()
plt.plot(np.arange(num_steps), mean_m_vector)
plt.xlabel('azimuth difference')
plt.ylabel('m')
plt.legend()
plt.grid()

# # # MSC
plt.figure()
plt.plot(np.arange(num_steps), mean_msc_vector)
plt.xlabel('azimuth difference')
plt.ylabel('MSC')
plt.legend()
plt.grid()

# # # MSW
plt.figure()
plt.plot(np.arange(num_steps), mean_msw_vector)
plt.xlabel('azimuth difference')
plt.ylabel('MSW')
plt.legend()
plt.grid()

# # # PSI
plt.figure()
plt.plot(np.arange(num_steps), mean_psi_vector, label='psi')
plt.plot(np.arange(num_steps), 1 - mean_ita_vector, label='ita')
plt.xlabel('azimuth difference')
plt.ylabel('$\Psi$')
plt.grid()
plt.legend()
plt.show()