import librosa
import numpy as np
import pyaudio
import matplotlib.pyplot as plt


##########################################################################################
##### IMPULSE RESPONSE

def generate_late_reverberation_ir_from_rt60(start_time, rt60, length, fs):
    """
    Returns a direct path (delta) at 0 delay plus a late reverberation
    in form of gaussian noise, with exponential decay.

    :param start_time: Time offset until the late reverb starts.
    :param rt60: rt60 time, in seconds
    :param length: length in seconds
    :param fs: sample rate
    :return: ir

    Example
    ________________
    start_time = 0.1
    rt60 = 1.5
    length = 2
    fs = 16000
    ir = generate_late_reverberation_ir_from_rt60(start_time, rt60, length, fs)

    plt.plot(np.arange(length*fs)/fs,ir)
    plt.grid()
    plt.show()

    plt.plot(np.arange(length*fs)/fs,10*np.log10(np.abs(ir)))
    plt.vlines(rt60,-100,0)
    plt.grid()
    plt.show()
    """

    num_samples = int(np.ceil(length * fs))
    start_sample = int(np.ceil(start_time * fs))
    n = np.arange(num_samples)
    num_samples_late_reverb = num_samples - start_sample

    ir = np.zeros(num_samples)
    ir[0] = 1.
    mean = 0.
    std = 1./3
    a = -np.log(1e-3)/rt60  # relates decay amplitude with rt60
    ir[start_sample:] = np.exp(-a*n[start_sample:]/fs) * np.random.normal(mean, std, num_samples_late_reverb)

    return ir


#### DRR habets, GANNOT (late reverberant spectral variance... ,eq. 23)
"""
est_DRR(k) = 10 log10 ( ( sum[n=0, nd-1] h(n)**2 ) / ( sum[nd, inf] h(n)**2 ) )
"""

def direct_reverberant_ratio(h, critical_time, fs):

    c = int(np.ceil(critical_time*fs))
    return np.sum(np.power(h[:c],2)) / np.sum(np.power(h[c:],2))


def decay_rate(t60, fs):
    """
    alpha, Bran Eq. 2.9
    :return:
    """
    return 3 * np.log(10) / ( t60 * fs )

def get_rt60(decay_rate, fs):
    """
    :return:
    """
    return 3 * np.log(10) / ( decay_rate * fs )


##########################################################################################
##### OTHER STUFF

def herm(A):
    return np.conj(np.transpose(A))

def play(ndarray, fs):
    """

    :param ndarray:
    :return:
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=fs,
                    output=True)

    stream.write(ndarray.astype(np.float32).tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()

def plot_magnitude_spectrogram(X, f, t, vmin=None, vmax=None):

    plt.pcolormesh(t, f, 20*np.log10(np.abs(X)), vmin=vmin, vmax=vmax)
    plt.ylim(0,20000)
    plt.yscale('symlog')
    plt.colorbar()


##########################################################################################
##### PSD LATE REVERB ESTIMATION


def dd_wiener_filter(Y, PSD_R, PSD_V, beta = 0.98):
    """
    :return:
    """
    K, N = Y.shape
    est_S = np.zeros((K, N), dtype='complex')
    WF = np.zeros((K, N))

    for n in range(1, N):
        # a priori SIR
        SIR_prio = (np.power(np.abs(est_S[:,n-1]), 2)) / (PSD_R[:,n-1] + PSD_V[:,n-1])
        # a posteriori SIR
        SIR_post = (np.power(np.abs(Y[:,n]), 2)) / (PSD_R[:,n] + PSD_V[:,n])
        # SIR estimate (3.7)
        est_SIR = beta * SIR_prio + (1-beta) * np.maximum(SIR_post-1, 0)
        # Wiener Filter (3.6)
        WF[:,n] = est_SIR / (est_SIR + 1)
        # est_S (3.4)
        est_S[:,n] = WF[:,n] * Y[:,n]

    return WF, est_S




def forward_exponential_habets(alpha, N, D, DRR, PSD_Y, PSD_V):
    """
    Estimate late reverb PSD based on exponential model
    Late reverberant...
    S. Bran Eq. 4.3

    alpha = reverb decay rate
    N = window hop (in samples)
    D = number of windows to start late reverb
    DRR = direct reverberant ratio
    psd_Y = psd of recorded signal
    psd_V= psd of noise signal

    kappa = psd_R / psd_D
    Only valid assumption if psd_D > psd_R (0 < kappa < 1)
    If psd_R > psd_D, then the first part is dropped and result is Lebart's estimator.
    """

    E = np.exp(-2*alpha*N*D)
    kappa = min( (1 - E) / (DRR * E ), 1)
    psd_X = np.maximum(PSD_Y - PSD_V, 0)

    K, N = PSD_Y.shape
    PSD_R = np.empty((K,N))
    for n in range(D,N):
        PSD_R[:,n] = ( (1-kappa) * E * PSD_R[:,n-D] ) + ( kappa * E * psd_X[:,n-D] )

    return PSD_R



##########################################################################################
##### MAR MODELS

def unwrap_MAR_coefs(c, L):
    """
    Take the coefs in vector form (c) as given by dereverberation_* methods,
    and rearrange them in matrix form (C)

    :param c: matrix, (Lc, dimK, dimN), Lc =  dimM * dimM * L
    :return: C: matrix(L, dimM, dimM, dimK, dimN). Most recent coefs first (D, D+1... Lar)
    """
    # todo assert dims

    Lc, dimK, dimN = c.shape
    dimM = int(np.sqrt(Lc/L))

    C = np.empty((L, dimM, dimM, dimK, dimN), dtype='complex')  # t, rows, cols
    for col in range(dimM):
        col_idx = col * L * dimM
        for l in range(L):
            l_idx = l * dimM + col_idx
            C[L - l - 1, col] = c[l_idx:l_idx + dimM]
    return C

def apply_MAR_coefs(C, s_tf, L, D, time_average=False):
    """

    :param C: matrix(L, dimM, dimM, dimK, dimN)
    :param s_tf:
    :return:
    """

    dimM, dimK, dimN = s_tf.shape
    r_tf = np.zeros(np.shape(s_tf), dtype='complex')

    if time_average: # only take coefficients from D on, although difference is very small (around -70 db for 3 seconds average)
        C = np.mean(C[:,:,:,:,D:], axis=-1)

    for n in range(D, dimN):
        for l in range(0, L):
            if time_average:
                a = np.transpose(C[l], (2, 0, 1))  # Expansion at the first dimension (k)
            else:
                a = np.transpose(C[l, :, :, :, n], (2, 0, 1))  # Expansion at the first dimension (k)
            b = r_tf[:, :, n - (D + l)].T[:, :, np.newaxis]  # Add a dummy dimension for the matmul
            r_tf[:, :, n] += (a @ b).squeeze().T  # Remove the dummy dimmension
        r_tf[:, :, n] += s_tf[:, :, n]  # Add novelty signal

    return r_tf

    # non-optimized version
    # r3_tf = np.zeros(np.shape(s_tf), dtype='complex')
    # for k in range(dimK):
    #     # print(k)
    #     for n in range(D, dimN):
    #         for l in range(0, L):
    #             r3_tf[:, k, n] += c[l, :, :, k] @ r3_tf[:, k, n-(D+l)]
    #         r3_tf[:, k, n] += s2_tf[:, k, n] # current
    # assert np.allclose(r2_tf,r3_tf)


def get_MAR_transition_matrix_eigenvalues(C, time_average=False):
    """

    https://dsp.stackexchange.com/questions/31859/how-do-i-test-stability-of-a-mimo-system

    :param C: matrix(L, dimM, dimM, dimK, dimN)
    :return:
    """

    L, dimM, _, dimK, dimN = C.shape

    if not time_average:
        e = np.empty((dimM * L, dimK, dimN), dtype=complex)

        for n in range(dimN):
            A = np.zeros((dimK, dimM * L, dimM * L), dtype=complex)
            # Put C matrices in the first M rows
            for l in range(L):
                A[:, :dimM, l * dimM:(l + 1) * dimM] = np.transpose(C[l, :, :, :, n], (2, 0, 1))  # put k in the first dim for broadcasting
            # Fill the resting M(L-1) x ML matrix with identity matrices
            I = np.tile(np.identity(dimM * L)[np.newaxis], (dimK, 1, 1))  # dimK identity matrices in parallel
            A[:, dimM:, :] = I[:, :-dimM]
            # Find eigenvalues
            e[:, :, n] = np.linalg.eigvals(A).T  # (dimM * L, dimK)
    else:
        C = np.mean(C, axis=-1)
        A = np.zeros((dimK, dimM * L, dimM * L), dtype=complex)  # Transition matrix
        # Put C matrices in the first M rows
        for l in range(L):
            A[:, :dimM, l * dimM:(l + 1) * dimM] = np.transpose(C[l], (2, 0, 1)) # put k in the first dim for broadcasting
        # Fill the resting M(L-1) x ML matrix with identity matrices
        I = np.tile(np.identity(dimM * L)[np.newaxis], (dimK, 1, 1))  # dimK identity matrices in parallel
        A[:, dimM:, :] = I[:, :-dimM]
        # Find eigenvalues
        e = np.linalg.eigvals(A).T  # (dimM * L, dimK)
    return e


# def todo_get_MAR_transition_matrix_eigenvalues(C, time_average=False):
#     """
#
#     :param C: matrix(L, dimM, dimM, dimK, dimN)
#     :return:
#     """
#
#     L, dimM, _, dimK, dimN = C.shape
#
#     if not time_average:
#         e = np.empty((dimM * L, dimK, dimN), dtype=complex)
#
#         for n in range(dimN):
#             A = np.zeros((dimK, dimM * L, dimM * L), dtype=complex)
#             # Put C matrices in the first M rows
#             for l in range(L):
#                 A[:, :dimM, l * dimM:(l + 1) * dimM] = np.transpose(C[l, :, :, :, n], (2, 0, 1))  # put k in the first dim for broadcasting
#             # Fill the resting M(L-1) x ML matrix with identity matrices
#             I = (np.identity(dimM * L)[np.newaxis, :]).repeat(dimK, axis=0)  # dimK identity matrices in parallel
#             A[:, dimM:, :] = I[:, :-dimM]
#             # Find eigenvalues
#             e[:, :, n] = np.linalg.eigvals(A).T  # (dimM * L, dimK)
#
#     else:
#         C = np.mean(C[:,:,:,:,D:], axis=-1)
#         A = np.zeros((dimK, dimM * L, dimM * L), dtype=complex)  # Transition matrix
#         # Put C matrices in the first M rows
#         for l in range(L):
#             A[:, :dimM, l * dimM:(l + 1) * dimM] = np.transpose(C[l], (2, 0, 1)) # put k in the first dim for broadcasting
#         # Fill the resting M(L-1) x ML matrix with identity matrices
#         I = (np.identity(dimM * L)[np.newaxis,:]).repeat(dimK, axis=0)  # dimK identity matrices in parallel
#         A[:, dimM:, :] = I[:,:-dimM]
#         # Find eigenvalues
#         e = np.linalg.eigvals(A).T  # (dimM * L, dimK)
#         # print(np.abs(e))
#     return e


def build_recursive_matrix(y, n, dimM, L):
    """
    operation described in Eq. 6.8
    :return:
    """

    # Construct y_Vec
    y_vec = np.zeros(dimM * L, dtype='complex')  # second term of # 6.8
    for x in np.arange(L):  # check!
        nd = L - 1 - x
        for m in range(dimM):
            y_vec[x * dimM + m] = y[m, n-nd]

    return np.kron(np.identity(dimM), y_vec)  # 6.8

# def todo_build_recursive_matrix(y, n, L):
#     """
#     operation described in Eq. 6.8
#     :return:
#     """
#
#     dimM, dimK, dimN = y.shape
#     Lc = dimM*dimM*L
#
#     # Construct y_Vec
#     y_vec = np.zeros((dimK, dimM * L), dtype='complex')  # second term of # 6.8
#     for x in np.arange(L):  # check!
#         nd = L - 1 - x
#         for m in range(dimM):
#             y_vec[:, x * dimM + m] = y[m, :, n-nd]
#
#     # TODO
#     out = np.zeros((dimK, dimM, Lc), dtype='complex')
#     for k in range(dimK):
#         out[k] = np.kron(np.identity(dimM), y_vec[k])  # 6.8
#     return out




def dereverberation_MAR(y_tf, D, L, alpha, ita):
    """
    Section 6.3

    :param y_tf:
    :param dimM:
    :param dimN:
    :param dimK:
    :param Lc:
    :param Lar:
    :param D:
    :param alpha:
    :param ita:
    :return:
    """

    dimM, dimK, dimN = y_tf.shape
    Lc = dimM * dimM * L

    est_s_tf = np.zeros((dimM, dimK, dimN), dtype='complex')
    est_c_tf = np.zeros((Lc, dimK, dimN), dtype='complex')

    for k in range(dimK):
        print(k)

        Y = np.zeros((dimN, dimM, Lc), dtype='complex')
        post_phi_s = np.zeros((dimN, dimM, dimM), dtype='complex')
        est_s = np.zeros((dimN, dimM), dtype='complex')

        est_c = np.zeros((dimN, Lc), dtype='complex')

        phi_c = np.zeros((dimN, Lc, Lc), dtype='complex')
        for n in range(D):
            phi_c[n] = np.identity(Lc)

        phi_s = np.zeros((dimN, dimM, dimM), dtype='complex')

        for n in range(D, dimN):
            # Build Y matrix, as 6.8
            Y[n] = build_recursive_matrix(y_tf[:, k, :], n, dimM, L)

            # Get estimates and covariance of regression coefs
            est_c[n], phi_c[n], phi_s[n] = estimate_MAR_regression_coefs(est_c[n-1], phi_c[n-1],
                                                               y_tf[:,k,n], Y[n - D],
                                                               alpha, Lc, ita,
                                                               post_phi_s[n - 1])

            # Get signal estimate
            est_s[n] = y_tf[:, k, n] - (Y[n - D] @ est_c[n])  # 6.9

            # Update posteriori signal covariance matrix for next iteration
            post_phi_s[n] = alpha * post_phi_s[n - 1] + (1 - alpha) * np.outer(est_s[n], herm(est_s[n]))  # 6.32 # TODO

            # Save data
            est_s_tf[:, k, :] = est_s.T
            est_c_tf[:, k, :] = est_c.T

    return est_s_tf, est_c_tf


def estimate_MAR_regression_coefs(last_c, last_phi_c, y, Y_D, alpha, Lc, ita, last_post_phi_s):

    # Estimate MAR coefficients
    ## A (transition matrix) is identity matrix (not written)
    ## phi_w (covariance of state perturbation noise) = var_w * I(Lc)
    # TODO FIX THE ZERO
    var_w = (1 / Lc) * np.mean(np.power(np.linalg.norm(0 - last_c), 2)) + ita  # 6.24
    phi_w = var_w * np.identity(Lc)

    # Estimate prediction error
    pred_phi_c = last_phi_c + phi_w  # 6.18
    pred_c = last_c  # 6.19
    e = y - (Y_D @ pred_c)  # 6.20

    # Braun method for target signal covariance
    phi_s = alpha * last_post_phi_s + (1 - alpha) * np.outer(e, herm(e))  # 6.31

    # Compute estimates of the MAR regression coefs
    K = (pred_phi_c @ herm(Y_D)) @ np.linalg.pinv(
        Y_D @ pred_phi_c @ herm(Y_D) + phi_s)  # 6.21
    phi_c = (np.identity(Lc) - K @ Y_D) @ pred_phi_c  # 6.22
    est_c = pred_c + K @ e  # 6.23

    return est_c, phi_c, phi_s

# def todo_dereverberation_MAR(y_tf, D, L, alpha, ita):
#     """
#     Section 6.3
#
#     :param y_tf:
#     :param dimM:
#     :param dimN:
#     :param dimK:
#     :param Lc:
#     :param Lar:
#     :param D:
#     :param alpha:
#     :param ita:
#     :return:
#     """
#
#     dimM, dimK, dimN = y_tf.shape
#     Lc = dimM * dimM * L
#
#     est_s_tf = np.zeros((dimM, dimK, dimN), dtype='complex')
#     est_c_tf = np.zeros((Lc, dimK, dimN), dtype='complex')
#
#
#     Y = np.zeros((dimK, dimN, dimM, Lc), dtype='complex')
#     post_phi_s = np.zeros((dimK, dimN, dimM, dimM), dtype='complex')
#     est_s = np.zeros((dimK, dimN, dimM), dtype='complex')
#
#     est_c = np.zeros((dimK, dimN, Lc), dtype='complex')
#
#     phi_c = np.zeros((dimK, dimN, Lc, Lc), dtype='complex')
#     for n in range(D):
#         phi_c[:, n] = np.tile(np.identity(Lc)[np.newaxis], (dimK, 1, 1))
#
#     phi_s = np.zeros((dimK, dimN, dimM, dimM), dtype='complex')
#
#     for n in range(D, dimN):
#         # Build Y matrix, as 6.8
#         Y[:, n] = todo_build_recursive_matrix(y_tf, n, L)
#
#         # Get estimates and covariance of regression coefs
#         for k in range(dimK):
#             est_c[k, n], phi_c[k, n], phi_s[k, n] = estimate_MAR_regression_coefs(est_c[:, n - 1], phi_c[:, n - 1],
#                                                                          y_tf[:, :, n], Y[:, n - D],
#                                                                          alpha, Lc, ita,
#                                                                          post_phi_s[:, n - 1])
#
#         # Get signal estimate
#         est_s[:, n] = y_tf[:, :, n] - ( (Y[:, n - D] @ est_c[:, n][:,np.newaxis]).squeeze() )  # 6.9
#
#         # Update posteriori signal covariance matrix for next iteration
#         # np.matmul(e[:,np.newaxis], np.conj(e[np.newaxis,:]) ) )
#         for k in range(dimK):
#             post_phi_s[k, n] = alpha * post_phi_s[k, n - 1] + (1 - alpha) * np.outer(est_s[k, n], herm(est_s[k, n]))  # 6.32 # TODO
#
#         # Save data
#         est_s_tf[:, :, :] = est_s.T
#         est_c_tf[:, :, :] = est_c.T
#
#     return est_s_tf, est_c_tf


# def todo_estimate_MAR_regression_coefs(last_c, last_phi_c, y, Y_D, alpha, Lc, ita, last_post_phi_s):
#
#     dimK, _ = last_c.shape
#     # Estimate MAR coefficients
#     ## A (transition matrix) is identity matrix (not written)
#     ## phi_w (covariance of state perturbation noise) = var_w * I(Lc)
#     # TODO FIX THE ZERO
#     var_w = (1 / Lc) * np.mean(np.power(np.linalg.norm(0 - last_c), 2)) + ita  # 6.24
#     phi_w = var_w * np.tile(np.identity(Lc)[np.newaxis], (dimK, 1, 1))
#
#     # Estimate prediction error
#     pred_phi_c = last_phi_c + phi_w  # 6.18
#     pred_c = last_c  # 6.19
#     e = y - (Y_D @ pred_c[:, :, np.newaxis]).squeeze()  # 6.20
#
#     # Braun method for target signal covariance
#     phi_s = alpha * last_post_phi_s + (1 - alpha) * np.outer(e, herm(e))  # 6.31
#
#     # Compute estimates of the MAR regression coefs
#     K = (pred_phi_c @ herm(Y_D)) @ np.linalg.pinv(
#         Y_D @ pred_phi_c @ herm(Y_D) + phi_s)  # 6.21
#     phi_c = (np.identity(Lc) - K @ Y_D) @ pred_phi_c  # 6.22
#     est_c = pred_c + K @ e  # 6.23
#
#     return est_c, phi_c, phi_s





def dereverberation_RLS(y_tf, D, L, alpha, alpha_RLS):
    """
    section 6.3.5
    :param y_tf:
    :param dimM:
    :param dimN:
    :param dimK:
    :param Lc:
    :param Lar:
    :param D:
    :param alpha:
    :param ita:
    :return:
    """

    dimM, dimK, dimN = y_tf.shape
    Lc = dimM * dimM * L

    est_s_tf = np.empty((dimM, dimK, dimN), dtype='complex')
    est_c_tf = np.empty((Lc, dimK, dimN), dtype='complex')

    for k in range(dimK):
        print(k)

        Y = np.zeros((dimN, dimM, Lc), dtype='complex')
        post_phi_s = np.zeros((dimN, dimM, dimM), dtype='complex')
        est_s = np.zeros((dimN, dimM), dtype='complex')

        est_c = np.zeros((dimN, Lc), dtype='complex')

        phi_c = np.zeros((dimN, Lc, Lc), dtype='complex')
        for n in range(D):
            phi_c[n] = np.identity(Lc)

        phi_s = np.zeros((dimN, dimM, dimM), dtype='complex') # Braun's covariance matrix
        rls_phi_s = np.zeros((dimN, dimM, dimM), dtype='complex') # RLS's covariance matrix


        for n in range(D, dimN):
            # Build Y matrix, as 6.8
            Y[n] = build_recursive_matrix(y_tf[:, k, :], n, dimM, L)

            # Get estimates and covariance of regression coefs
            est_c[n], phi_c[n], phi_s[n], rls_phi_s[n] = estimate_RLS_regression_coefs(est_c[n - 1], phi_c[n - 1],
                                                                                       y_tf[:, k, n], Y[n - D],
                                                                                       alpha, Lc, alpha_RLS,
                                                                                       post_phi_s[n - 1])
            # Get signal estimate
            est_s[n] = y_tf[:, k, n] - (Y[n - D] @ est_c[n])  # 6.9

            # Update posteriori signal covariance matrix for next iteration
            post_phi_s[n] = alpha * post_phi_s[n - 1] + (1 - alpha) * np.outer(est_s[n], herm(est_s[n]))  # 6.32 # TODO

            # Save data
            est_s_tf[:, k, :] = est_s.T
            est_c_tf[:, k, :] = est_c.T

    return est_s_tf, est_c_tf


def estimate_RLS_regression_coefs(last_c, last_phi_c, y, Y_D, alpha, Lc, alpha_RLS, last_post_phi_s):


    # Estimate RLS coefficients

    # Estimate prediction error
    pred_phi_c = (1/alpha_RLS) * last_phi_c  # 6.33 (instead of 6.18)
    pred_c = last_c  # 6.19
    e = y - (Y_D @ pred_c)  # 6.20

    ## Braun method for phi_s
    phi_s = alpha * last_post_phi_s + (1 - alpha) * np.outer(e, herm(e))  # 6.31
    ## RLS method
    # in RLS, phi_s is diagonal (no correlation among channels)
    # but power has to be estimated anyway.
    # So, as proposed in 6.7.2, let's scale an identity matrix
    # by the mean of the main diagonal of the Braun's covariance matrix (our phi_s)
    rls_phi_S = np.mean(np.diag(phi_s)) * np.identity(np.size(y))

    # Compute estimates of the MAR regression coefs
    K = (pred_phi_c @ herm(Y_D)) @ np.linalg.pinv(
        Y_D @ pred_phi_c @ herm(Y_D) + rls_phi_S)  # 6.21
    phi_c = (np.identity(Lc) - K @ Y_D) @ pred_phi_c  # 6.22
    est_c = pred_c + K @ e  # 6.23

    return est_c, phi_c, phi_s, rls_phi_S




# def dereverberation_MAR_oracle(y_tf, dimM, dimN, dimK, Lc, Lar, D, alpha, ita, s):
def dereverberation_MAR_oracle(y_tf, D, L, alpha, ita, s):
    """
    section 6.3.5
    :param y_tf:
    :param dimM:
    :param dimN:
    :param dimK:
    :param Lc:
    :param Lar:
    :param D:
    :param alpha:
    :param ita:
    :param s: true target signal
    :return:
    """

    dimM, dimK, dimN = y_tf.shape
    Lc = dimM * dimM * L

    est_s_tf = np.empty((dimM, dimK, dimN), dtype='complex')
    est_c_tf = np.empty((Lc, dimK, dimN), dtype='complex')

    for k in range(dimK):
        print(k)

        Y = np.zeros((dimN, dimM, Lc), dtype='complex')
        est_s = np.zeros((dimN, dimM), dtype='complex')

        est_c = np.zeros((dimN, Lc), dtype='complex')

        phi_c = np.zeros((dimN, Lc, Lc), dtype='complex')
        for n in range(D):
            phi_c[n] = np.identity(Lc)

        oracle_phi_s = np.zeros((dimN, dimM, dimM), dtype='complex') # True covariance matrix

        for n in range(D, dimN):
            # Build Y matrix, as 6.8
            Y[n] = build_recursive_matrix(y_tf[:, k, :], n, dimM, L)

            oracle_phi_s[n] = alpha * oracle_phi_s[n - 1] + (1 - alpha) * np.outer(s[:, k, n], herm(s[:, k, n]))  # 6.32 # TODO

            # Get estimates and covariance of regression coefs
            est_c[n], phi_c[n] = estimate_MAR_oracle_regression_coefs(est_c[n - 1], phi_c[n - 1],
                                                                                       y_tf[:, k, n], Y[n - D],
                                                                                       Lc, ita,
                                                                                       oracle_phi_s[n])
            # Get signal estimate
            est_s[n] = y_tf[:, k, n] - (Y[n - D] @ est_c[n])  # 6.9

            # Save data
            est_s_tf[:, k, :] = est_s.T
            est_c_tf[:, k, :] = est_c.T

    return est_s_tf, est_c_tf


def estimate_MAR_oracle_regression_coefs(last_c, last_phi_c, y, Y_D, Lc, ita, oracle_phi_s):

    # Estimate MAR coefficients
    ## A (transition matrix) is identity matrix (not written)
    ## phi_w (covariance of state perturbation noise) = var_w * I(Lc)
    var_w = (1 / Lc) * np.mean(np.power(np.linalg.norm(0 - last_c), 2)) + ita  # 6.24
    phi_w = var_w * np.identity(Lc)

    # Estimate prediction error
    pred_phi_c = last_phi_c + phi_w  # 6.18
    pred_c = last_c  # 6.19
    e = y - (Y_D @ pred_c)  # 6.20

    # Compute estimates of the MAR regression coefs
    K = (pred_phi_c @ herm(Y_D)) @ np.linalg.pinv(
        Y_D @ pred_phi_c @ herm(Y_D) + oracle_phi_s)  # 6.21
    phi_c = (np.identity(Lc) - K @ Y_D) @ pred_phi_c  # 6.22
    est_c = pred_c + K @ e  # 6.23

    return est_c, phi_c






def inner_norm(d, PHI):
    return np.sqrt( herm(d) @ np.linalg.pinv(PHI) @ d )

def estimate_MAR_sparse(y_tf, L, tau, p, i_max, ita, epsilon):
    """
    GROUP SPARSITY FOR MIMO SPEECH DEREVERBERATION

    :return:
    """

    dimM, dimK, dimN = y_tf.shape

    est_s_tf = np.empty(np.shape(y_tf), dtype=complex)
    C = np.empty((dimK, dimM * L, dimM), dtype=complex)
    phi = np.empty((dimK, dimM, dimM), dtype=complex)

    for k in range(dimK):
        print(k)
        X = y_tf[:, k, :].T  # [N, M]
        i = 0
        D = X  # [N, M]
        PHI = np.identity(dimM)  # [M, M]
        F = ita  # just for initialization

        # Get recursive matrix
        Xtau = np.zeros((dimN, dimM * L), dtype=complex)  # [N, ML]
        for m in range(dimM):
            Xtau_m = np.zeros((dimN, L), dtype=complex)
            for l in range(L):
                for n in range(dimN):
                    if n >= tau + l:  # avoid aliasing
                        Xtau_m[n, l] = X[n - tau - l, m]
            Xtau[:, L * m:L * (m + 1)] = Xtau_m

        while i < i_max and F >= ita:
            print('  iter',i, 'F', F)
            last_D = D # [N, M]

            # Estimate weights
            w = np.empty(dimN, dtype='complex')  # [N]
            for n in range(dimN):
                d_n = last_D[n, :][:, np.newaxis] # [N,1]
                w[n] = np.power(np.power(inner_norm(d_n, PHI), 2) + epsilon, (p / 2) - 1)

            # Estimate G
            W = np.diag(w)  # [N, N]
            G = np.linalg.pinv(herm(Xtau) @ W @ Xtau) @ (herm(Xtau) @ W @ X)  # [ML, M]

            # Estimate D
            D = X - (Xtau @ G)

            # Estimate PHI
            PHI = (1 / dimN) * (D.T @ W @ D.conj())

            # Estimate convergence
            F = np.linalg.norm(D - last_D) / np.linalg.norm(D)

            # Update pointer
            i += 1

        # Assign
        est_s_tf[:, k, :] = D.T
        C[k, :, :] = G
        phi[k, :, :] = PHI

    return est_s_tf, C, phi


def estimate_MAR_sparse_parallel(y_tf, L, tau, p, i_max, ita, epsilon):
    """
    GROUP SPARSITY FOR MIMO SPEECH DEREVERBERATION

    :return:
    """

    dimM, dimK, dimN = y_tf.shape

    X = y_tf.transpose((1,2,0))  # [K, N, M]
    i = 0
    D = X  # [K, N, M]
    PHI = np.tile(np.identity(dimM)[np.newaxis], (dimK, 1, 1)) # [K, M, M]
    F = ita  # just for initialization
    F_k = ita  # just for initialization

    # Get recursive matrix
    Xtau = np.zeros((dimK, dimN, dimM * L), dtype=complex)  # [K, N, ML]
    for m in range(dimM):
        Xtau_m = np.zeros((dimK, dimN, L), dtype=complex) # [K, N, L]
        for l in range(L):
            for n in range(dimN):
                if n >= tau + l:  # avoid aliasing
                    Xtau_m[:, n, l] = X[:, n - tau - l, m]
        Xtau[:, :, L * m:L * (m + 1)] = Xtau_m

    # while i < i_max and F >= ita:
    while i < i_max and np.mean(F_k) >= ita:
        print('  iter',i, 'np.mean(F_k)', np.mean(F_k))

        last_D = D # [K, N, M]

        def herm_k(X):
            return X.conj().transpose((0, 2, 1))

        def transpose_k(X):
        def estimate_MAR_sparse_parallel(y_tf, L, tau, p, i_max, ita, epsilon):
            """
            GROUP SPARSITY FOR MIMO SPEECH DEREVERBERATION

            :return:
            """

            dimM, dimK, dimN = y_tf.shape

            X = y_tf.transpose((1,2,0))  # [K, N, M]
            i = 0
            D = X  # [K, N, M]
            PHI = np.tile(np.identity(dimM)[np.newaxis], (dimK, 1, 1)) # [K, M, M]
            F = ita  # just for initialization
            F_k = ita  # just for initialization

            # Get recursive matrix
            Xtau = np.zeros((dimK, dimN, dimM * L), dtype=complex)  # [K, N, ML]
            for m in range(dimM):
                Xtau_m = np.zeros((dimK, dimN, L), dtype=complex) # [K, N, L]
                for l in range(L):
                    for n in range(dimN):
                        if n >= tau + l:  # avoid aliasing
                            Xtau_m[:, n, l] = X[:, n - tau - l, m]
                Xtau[:, :, L * m:L * (m + 1)] = Xtau_m

            # while i < i_max and F >= ita:
            while i < i_max and np.mean(F_k) >= ita:
                print('  iter',i, 'np.mean(F_k)', np.mean(F_k))

                last_D = D # [K, N, M]

                def herm_k(X):
                    return X.conj().transpose((0, 2, 1))

                def transpose_k(X):
                    return X.transpose((0, 2, 1))

                # Estimate weights
                w = np.empty((dimK, dimN), dtype='complex')  # [K, N]
                for n in range(dimN):
                    d_n = last_D[:, n, :][:, :, np.newaxis]  # [K, N, 1]
                    # inner = np.squeeze(np.sqrt(d_n.conj().transpose((0, 2, 1)) @ np.linalg.pinv(PHI) @ d_n)) # [K]
                    inner = np.squeeze(np.sqrt(herm_k(d_n) @ np.linalg.pinv(PHI) @ d_n)) # [K]
                    w[:, n] = np.power(np.power(inner, 2) + epsilon, (p / 2) - 1)

                # Estimate G
                # todo parallelize
                W = np.empty((dimK, dimN, dimN), dtype=complex) # [K, N, N]
                for k in range(dimK):
                    W[k] = np.diag(w[k])

                G = np.linalg.pinv(herm_k(Xtau) @ W @ Xtau) @ (herm_k(Xtau) @ W @ X)  # [K, ML, M]

                # Estimate D
                D = X - (Xtau @ G) # [K, N, M]

                # Estimate PHI
                PHI = (1 / dimN) * (transpose_k(D) @ W @ D.conj()) # [K, M, M]

                # Estimate convergence
                # F = np.linalg.norm(D - last_D) / np.linalg.norm(D)
                # Per-band convergence
                F_k =  np.linalg.norm(D - last_D, axis=(1,2)) / np.linalg.norm(D, axis=(1,2))
                # print(F_k < ita_k)
                # print(F_k)
                # plt.figure()
                # plt.title(str(i))
                # plt.plot(F_k)
                # plt.hlines(np.mean(F_k), 0,dimK-1)
                # plt.show()
                # print(np.mean(F_k), F)

                # Update pointer
                i += 1

            return D.transpose((2, 0, 1)), G, PHI



        def estimate_MAR_sparse_oracle(y_tf, s_tf, L, tau, p, i_max, ita, epsilon):
            """
            GROUP SPARSITY FOR MIMO SPEECH DEREVERBERATION

            oracle SCM
            :return:
            """

            dimM, dimK, dimN = y_tf.shape

            est_s_tf = np.empty(np.shape(y_tf), dtype=complex)
            C = np.empty((dimK, dimM*L, dimM), dtype=complex)

            phi_oracle = np.empty((dimK, dimM, dimM), dtype=complex)
            for k in range(dimK):
                s = s_tf[:, k, :].T
                phi_oracle[k] = herm(s) @ s

            for k in range(dimK):
                print(k)
                X = y_tf[:, k, :].T  # [N, M]
                i = 0
                D = X # [N, M]
                F = ita # just for initialization

                # Get recursive matrix
                Xtau = np.zeros((dimN, dimM * L), dtype=complex)  # [N, ML]
                for m in range(dimM):
                    Xtau_m = np.zeros((dimN, L), dtype=complex)
                    for l in range(L):
                        for n in range(dimN):
                            if n >= tau + l :  # avoid aliasing
                                Xtau_m[n, l] = X[n - tau - l, m]
                    Xtau[:, L * m:L * (m + 1)] = Xtau_m

                while i < i_max and F >= ita:
                    # print('  iter',i, 'F', F)
                    last_D = D

                    # Estimate weights
                    w = np.empty(dimN, dtype='complex') # probably real...
                    for n in range(dimN):
                        d_n = last_D[n,:][:,np.newaxis]
                        w[n] = np.power( np.power(inner_norm(d_n, phi_oracle[k]), 2) + epsilon, (p/2)-1)

                    # Estimate G
                    W = np.diag(w) # [N, N]
                    G = np.linalg.pinv( herm(Xtau) @ W @ Xtau ) @ ( herm(Xtau) @ W @ X )  # [ML, M]

                    # Estimate D
                    D = X - ( Xtau @ G )

                    # Estimate convergence
                    F = np.linalg.norm(D - last_D) / np.linalg.norm(D)

                    # Update pointer
                    i += 1

                # Assign
                est_s_tf[:, k, :] = D.T
                C[k, :, :] = G

            return est_s_tf, C


        def estimate_MAR_sparse_identity(y_tf, L, tau, p, i_max, ita, epsilon):
            """
            GROUP SPARSITY FOR MIMO SPEECH DEREVERBERATION

            oracle SCM
            :return:
            """

            dimM, dimK, dimN = y_tf.shape

            est_s_tf = np.empty(np.shape(y_tf), dtype=complex)
            C = np.empty((dimK, dimM*L, dimM), dtype=complex)

            # phi as identity matrix
            phi = np.empty((dimK, dimM, dimM), dtype=complex)
            for k in range(dimK):
                phi[k] = np.identity(dimM)

            for k in range(dimK):
                print(k)
                X = y_tf[:, k, :].T  # [N, M]
                i = 0
                D = X # [N, M]
                F = ita # just for initialization

                # Get recursive matrix
                Xtau = np.zeros((dimN, dimM * L), dtype=complex)  # [N, ML]
                for m in range(dimM):
                    Xtau_m = np.zeros((dimN, L), dtype=complex)
                    for l in range(L):
                        for n in range(dimN):
                            if n >= tau + l :  # avoid aliasing
                                Xtau_m[n, l] = X[n - tau - l, m]
                    Xtau[:, L * m:L * (m + 1)] = Xtau_m

                while i < i_max and F >= ita:
                    # print('  iter',i, 'F', F)
                    last_D = D

                    # Estimate weights
                    w = np.empty(dimN, dtype='complex') # probably real...
                    for n in range(dimN):
                        d_n = last_D[n,:][:,np.newaxis]
                        w[n] = np.power( np.power(inner_norm(d_n, phi[k]), 2) + epsilon, (p/2)-1)

                    # Estimate G
                    W = np.diag(w) # [N, N]
                    G = np.linalg.pinv( herm(Xtau) @ W @ Xtau ) @ ( herm(Xtau) @ W @ X )  # [ML, M]

                    # Estimate D
                    D = X - ( Xtau @ G )

                    # Estimate convergence
                    F = np.linalg.norm(D - last_D) / np.linalg.norm(D)

                    # Update pointer
                    i += 1

                # Assign
                est_s_tf[:, k, :] = D.T
                C[k, :, :] = G

            return est_s_tf, C
            return X.transpose((0, 2, 1))

        # Estimate weights
        w = np.empty((dimK, dimN), dtype='complex')  # [K, N]
        for n in range(dimN):
            d_n = last_D[:, n, :][:, :, np.newaxis]  # [K, N, 1]
            # inner = np.squeeze(np.sqrt(d_n.conj().transpose((0, 2, 1)) @ np.linalg.pinv(PHI) @ d_n)) # [K]
            inner = np.squeeze(np.sqrt(herm_k(d_n) @ np.linalg.pinv(PHI) @ d_n)) # [K]
            w[:, n] = np.power(np.power(inner, 2) + epsilon, (p / 2) - 1)

        # Estimate G
        # todo parallelize
        W = np.empty((dimK, dimN, dimN), dtype=complex) # [K, N, N]
        for k in range(dimK):
            W[k] = np.diag(w[k])

        G = np.linalg.pinv(herm_k(Xtau) @ W @ Xtau) @ (herm_k(Xtau) @ W @ X)  # [K, ML, M]

        # Estimate D
        D = X - (Xtau @ G) # [K, N, M]

        # Estimate PHI
        PHI = (1 / dimN) * (transpose_k(D) @ W @ D.conj()) # [K, M, M]

        # Estimate convergence
        # F = np.linalg.norm(D - last_D) / np.linalg.norm(D)
        # Per-band convergence
        F_k =  np.linalg.norm(D - last_D, axis=(1,2)) / np.linalg.norm(D, axis=(1,2))
        # print(F_k < ita_k)
        # print(F_k)
        # plt.figure()
        # plt.title(str(i))
        # plt.plot(F_k)
        # plt.hlines(np.mean(F_k), 0,dimK-1)
        # plt.show()
        # print(np.mean(F_k), F)

        # Update pointer
        i += 1

    return D.transpose((2, 0, 1)), G, PHI



def estimate_MAR_sparse_oracle(y_tf, s_tf, L, tau, p, i_max, ita, epsilon):
    """
    GROUP SPARSITY FOR MIMO SPEECH DEREVERBERATION

    oracle SCM
    :return:
    """

    dimM, dimK, dimN = y_tf.shape

    est_s_tf = np.empty(np.shape(y_tf), dtype=complex)
    C = np.empty((dimK, dimM*L, dimM), dtype=complex)

    phi_oracle = np.empty((dimK, dimM, dimM), dtype=complex)
    for k in range(dimK):
        s = s_tf[:, k, :].T
        phi_oracle[k] = herm(s) @ s

    for k in range(dimK):
        print(k)
        X = y_tf[:, k, :].T  # [N, M]
        i = 0
        D = X # [N, M]
        F = ita # just for initialization

        # Get recursive matrix
        Xtau = np.zeros((dimN, dimM * L), dtype=complex)  # [N, ML]
        for m in range(dimM):
            Xtau_m = np.zeros((dimN, L), dtype=complex)
            for l in range(L):
                for n in range(dimN):
                    if n >= tau + l :  # avoid aliasing
                        Xtau_m[n, l] = X[n - tau - l, m]
            Xtau[:, L * m:L * (m + 1)] = Xtau_m

        while i < i_max and F >= ita:
            # print('  iter',i, 'F', F)
            last_D = D

            # Estimate weights
            w = np.empty(dimN, dtype='complex') # probably real...
            for n in range(dimN):
                d_n = last_D[n,:][:,np.newaxis]
                w[n] = np.power( np.power(inner_norm(d_n, phi_oracle[k]), 2) + epsilon, (p/2)-1)

            # Estimate G
            W = np.diag(w) # [N, N]
            G = np.linalg.pinv( herm(Xtau) @ W @ Xtau ) @ ( herm(Xtau) @ W @ X )  # [ML, M]

            # Estimate D
            D = X - ( Xtau @ G )

            # Estimate convergence
            F = np.linalg.norm(D - last_D) / np.linalg.norm(D)

            # Update pointer
            i += 1

        # Assign
        est_s_tf[:, k, :] = D.T
        C[k, :, :] = G

    return est_s_tf, C


def estimate_MAR_sparse_identity(y_tf, L, tau, p, i_max, ita, epsilon):
    """
    GROUP SPARSITY FOR MIMO SPEECH DEREVERBERATION

    oracle SCM
    :return:
    """

    dimM, dimK, dimN = y_tf.shape

    est_s_tf = np.empty(np.shape(y_tf), dtype=complex)
    C = np.empty((dimK, dimM*L, dimM), dtype=complex)

    # phi as identity matrix
    phi = np.empty((dimK, dimM, dimM), dtype=complex)
    for k in range(dimK):
        phi[k] = np.identity(dimM)

    for k in range(dimK):
        print(k)
        X = y_tf[:, k, :].T  # [N, M]
        i = 0
        D = X # [N, M]
        F = ita # just for initialization

        # Get recursive matrix
        Xtau = np.zeros((dimN, dimM * L), dtype=complex)  # [N, ML]
        for m in range(dimM):
            Xtau_m = np.zeros((dimN, L), dtype=complex)
            for l in range(L):
                for n in range(dimN):
                    if n >= tau + l :  # avoid aliasing
                        Xtau_m[n, l] = X[n - tau - l, m]
            Xtau[:, L * m:L * (m + 1)] = Xtau_m

        while i < i_max and F >= ita:
            # print('  iter',i, 'F', F)
            last_D = D

            # Estimate weights
            w = np.empty(dimN, dtype='complex') # probably real...
            for n in range(dimN):
                d_n = last_D[n,:][:,np.newaxis]
                w[n] = np.power( np.power(inner_norm(d_n, phi[k]), 2) + epsilon, (p/2)-1)

            # Estimate G
            W = np.diag(w) # [N, N]
            G = np.linalg.pinv( herm(Xtau) @ W @ Xtau ) @ ( herm(Xtau) @ W @ X )  # [ML, M]

            # Estimate D
            D = X - ( Xtau @ G )

            # Estimate convergence
            F = np.linalg.norm(D - last_D) / np.linalg.norm(D)

            # Update pointer
            i += 1

        # Assign
        est_s_tf[:, k, :] = D.T
        C[k, :, :] = G

    return est_s_tf, C
##########################################################################################
##### PLOT

def plot_signal(s, title=None):

    if s.ndim == 1:
        s = np.expand_dims(s, axis=0)
    dimM = np.shape(s)[0]

    plt.figure()
    if title is not None:
        plt.suptitle(title, y=1)

    if dimM == 1:
        plt.plot(s[0])
    else:
        for m in range(dimM):
            plt.subplot(2,2,m+1)
            plt.plot(s[m])


def plot_magnitude_spectrogram(tf, title=None):

    # if yscale == 'log':
        # tf = 20*np.log10(np.abs(tf))

    db_range = 60 # dB
    vmax = np.max(20 * np.log10(np.abs(tf)))
    vmin = vmax - db_range

    plt.figure()
    if title is not None:
        plt.suptitle(title, y=1)

    if tf.ndim == 2:
        plt.pcolormesh(20*np.log10(np.abs(tf)), vmin=vmin, vmax=vmax, cmap='inferno')
        plt.colorbar()
    else:
        dimM = tf.shape[0]
        if dimM == 1:
            plt.pcolormesh(20 * np.log10(np.abs(tf[0])), vmin=vmin, vmax=vmax, cmap='inferno')
            plt.colorbar()
        else:
            for m in range(dimM):
                plt.subplot(2,2,m+1)
                plt.pcolormesh(20*np.log10(np.abs(tf[m])), vmin=vmin, vmax=vmax, cmap='inferno')
                plt.colorbar()

def plot_phase_spectrogram(tf, title=None):

    plt.figure()
    if title is not None:
        plt.suptitle(title)

    if tf.ndim == 2:
        plt.pcolormesh(np.angle(tf), cmap='inferno')
        plt.colorbar()
    else:
        dimM = tf.shape[0]
        for m in range(dimM):
            plt.subplot(2,2,m+1)
            plt.pcolormesh(np.angle(tf[m]), cmap='inferno')
            plt.colorbar()



##########################################################################################
##### METRICS

# TODO CHECK BSD "An objective measure for predicting subjective quality of speech coders"


def log_spectral_difference(a, b):
    ## a,b are matrices of same shape
    a = np.abs(a)
    b = np.abs(b)
    a = a[b>1e-20]
    b = b[b>1e-20]
    ratio = np.abs(a/b)
    # remove by hand divide by zeros
    ratio = ratio[~np.isinf(ratio)]
    return np.sqrt(np.mean(np.power(20*np.log10(ratio), 2)))

def log_spectral_difference_improvement(Y, S, est_S):
    """
    MUSIC DEREVERBERATION USING HARMONIC STRUCTURE SOURCE MODEL AND WIENER FILTER
    :param Y:
    :param S:
    :param est_S:
    :return:
    """
    return log_spectral_difference(Y, S) - log_spectral_difference(est_S, S)

# def signal_interference_ratio:
#     """
#     from Online Dereverberation for Dynamic Scenarios Using a Kalman Filter With an Autoregressive Model
#
#     "and the signal - to - interference ratio(SIR), where the interference is the residual reverberation plus artifacts"
#
#     :return:
#     """"""
#

def segmental_SRR(true_s, est_s, fs, h=0.02):
    """
    "A DEREVERBERATION ALGORITHM FOR SPHERICAL MICROPHONE ARRAYS USING COMPRESSED SENSING TECHNIQUES"
    h = window size in seconds
    :return: 
    """

    def NSRR(true_s, est_s):
        gamma = np.linalg.lstsq(true_s[:, np.newaxis], est_s[:, np.newaxis], rcond=None)[0]

        return 10 * np.log10(
            np.power(np.linalg.norm(gamma * true_s), 2) / np.power(np.linalg.norm(est_s - gamma * true_s), 2))


    hop_samples = int(np.floor(h*fs))
    num_windows = int(np.ceil(true_s.size/hop_samples))
    nsrr = np.empty(num_windows)
    for nw in range(num_windows):
        start = nw*hop_samples
        end = (nw+1)*hop_samples
        nsrr[nw] = NSRR(true_s[start:end], est_s[start:end])

    return nsrr



def log_PSD_error(ref_PSD, est_PSD, D=0):
    '''
    Braun, Eq B.8
    :return:
    '''

    e = 10*np.log10(est_PSD[:,D:]/ref_PSD[:,D:])
    mean_e = np.mean(e)

    lower_e = e[e<=mean_e]
    lower_semi_variance_e = (1/lower_e.size) * np.sum(np.power(lower_e-mean_e,2))

    upper_e = e[e>mean_e]
    upper_semi_variance_e = (1/upper_e.size) * np.sum(np.power(upper_e-mean_e,2))

    return e, mean_e, lower_semi_variance_e, upper_semi_variance_e



#
# def cepsdist(x, y, fs, frame, shift):
#     """
#     function [d, e] = cepsdist(x, y, fs, param);
#     %% CEPSDIST
#     %% Cepstral distance between two signals
#     %%
#     %% [D, E] = CEPSDIST(X, Y, FS, PARAM) calculates cepstral distance between
#     %% two one-dimensional signals specified by X and Y.
#     %%
#     %% Written and distributed by the REVERB challenge organizers on 1 July, 2013
#     %% Inquiries to the challenge organizers (REVERB-challenge@lab.ntt.co.jp)
#
#     :param x:
#     :param y:
#     :param fs:
#     :param frame: in seconds
#     :param shift: in seconds
#
#     :return:
#     """
#
# # % Calculate the number of frames.
# # %----------------------------------------------------------------------
#     if x.size > y.size:
#         x = x[:y.size]
#     else
#         y = y[x.size]
#
# # %% Normalization
# # if ~strcmp(param.cmn, 'y')
#     x = x / np.sqrt(np.sum(np.power(x,2)))
#     y = y / np.sqrt(np.sum(np.power(y,2)))
#
#
#     frame = int(np.fix(frame * fs))
#     shift = int(np.fix(shift * fs))
#
#     num_sample = x.size
#     num_frame  = np.fix((num_sample - frame + shift) / shift)
#
#
# # % Break up the signals into frames.
# # %----------------------------------------------------------------------
#
# win = window(param.window, frame);
#
# idx = repmat((1 : frame)', 1, num_frame) + ...
#       repmat((0 : num_frame - 1) * shift, frame, 1);
#
# X = bsxfun(@times, x(idx), win);
# Y = bsxfun(@times, y(idx), win);
#
#
# % Apply the cepstrum analysis.
# %----------------------------------------------------------------------
#
# ceps_x = realceps(X);
# ceps_y = realceps(Y);
#
# ceps_x = ceps_x(1 : param.order + 1, :);
# ceps_y = ceps_y(1 : param.order + 1, :);
#
#
# % Perform cepstral mean normalization.
# %----------------------------------------------------------------------
#
# if strcmp(param.cmn, 'y')
#   ceps_x = bsxfun(@minus, ceps_x, mean(ceps_x, 2));
#   ceps_y = bsxfun(@minus, ceps_y, mean(ceps_y, 2));
# end
#
#
# % Calculate the cepstral distances
# %----------------------------------------------------------------------
#
# err = (ceps_x - ceps_y) .^2;
# ds  = 10 / log(10) * sqrt(2 * sum(err(2 : end, :), 1) + err(1, :));
# ds  = min(ds, 10);
# ds  = max(ds, 0);
#
# d = mean(ds);
# e = median(ds);
#
