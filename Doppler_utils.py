import numpy as np
from numba import njit

def generate_delay_Doppler_channel_parameters(
    N, M, f_c, delta_f, T, max_speed, profile='NTN-TDL-A'
):
    """
    Generates channel parameters (channel coefficients, delay taps, Doppler taps)
    based on 3GPP TR 38.811 Tables 6.9.2-1 to 6.9.2-4 for NTN-TDL-A/B/C/D at elevation 50°.

    In this version, a single 'bulk' Doppler offset is applied to all taps, as
    typically assumed in TR 38.811 for LEO satellite channels.

    Parameters:
        N         : Number of OFDM symbols (affects Doppler resolution)
        M         : Number of subcarriers (affects delay resolution)
        f_c       : Carrier frequency in Hz (e.g., 2e9 for 2 GHz)
        delta_f   : Subcarrier spacing in Hz (e.g., 15e3)
        T         : OFDM symbol duration in seconds (including CP)
        max_speed : Maximum relative speed in km/h (LEO or user speed)
        profile   : 'NTN-TDL-A', 'NTN-TDL-B', 'NTN-TDL-C', or 'NTN-TDL-D'

    Returns:
        chan_coef    : complex channel coefficients for each tap (numpy array)
        delay_taps   : integer array of delay-tap indices
        doppler_taps : integer array of Doppler-tap indices (all identical if truly bulk)
        taps         : number of channel taps
    """

    # -------------------------------------------------------
    # 1) Define delay (µs) and power (dB) from TR 38.811 at 50° elevation
    # -------------------------------------------------------
    if profile == 'NTN-TDL-A':
        # Table 6.9.2-1: 3 taps, all Rayleigh
        # Normalized Delay (µs), Power (dB)
        delays_us = np.array([0, 1.0811, 2.8416])
        pdp_db    = np.array([0, -4.675, -6.482])
        K_dB = None
    elif profile == 'NTN-TDL-B':
        # Table 6.9.2-2: 4 taps, all Rayleigh
        delays_us = np.array([0, 0.7249, 0.7410, 5.7392])
        pdp_db    = np.array([0, -1.973, -4.332, -11.914])
        K_dB = None
    elif profile == 'NTN-TDL-C':
        # Table 6.9.2-3: 2 taps; first factor is LOS path with K-factor=10.224 dB
        delays_us = np.array([0, 0, 14.8124])
        pdp_db    = np.array([-0.394, -10.618, -23.373])
        # Note from Table: first tap has K=10.224 dB
        K_dB = 10.224
    elif profile == 'NTN-TDL-D':
        # Table 6.9.2-4: 3 taps; first factor is LOS path with K-factor=11.707 dB
        delays_us = np.array([0, 0, 0.5596, 7.3340])
        pdp_db    = np.array([-0.284, -11.991, -9.887, -16.771])
        # Note from Table: first tap has K=11.707 dB
        K_dB = 11.707
    else:
        raise ValueError("profile must be one of 'NTN-TDL-A', 'NTN-TDL-B', 'NTN-TDL-C', 'NTN-TDL-D'")

    # -------------------------------------------------------
    # 2) Convert from µs to seconds, compute total # of taps
    # -------------------------------------------------------
    delays_s = delays_us * 1e-6
    taps     = len(delays_s)

    # -------------------------------------------------------
    # 3) Compute delay indices for your simulation grid
    #    - "one_delay_tap" is the time resolution for each discrete delay bin
    # -------------------------------------------------------
    one_delay_tap = 1 / (M * delta_f)  # seconds per delay bin
    delay_taps = np.rint(delays_s / one_delay_tap).astype(int)

    # -------------------------------------------------------
    # 4) Convert dB powers -> linear scale, then normalize so sum = 1
    # -------------------------------------------------------
    p_lin = 10 ** (pdp_db / 10)
    p_sum = np.sum(p_lin)
    p_lin_norm = p_lin / p_sum

    # -------------------------------------------------------
    # 5) Generate channel coefficients for each tap
    #    - TDL-C/D: first tap is LOS (Ricean), others are Rayleigh
    #    - TDL-A/B: all taps are Rayleigh
    # -------------------------------------------------------
    chan_coef = np.zeros(taps, dtype=complex)

    if K_dB is not None:
        # LOS scenario: the first tap has a K-factor
        K_lin = 10 ** (K_dB / 10)  # linear K-factor
        # Power allocated to the first tap
        p_first = p_lin_norm[0]
        # For a Ricean tap with total average power p_first:
        #  - LOS amplitude^2 = p_first * (K_lin/(K_lin+1))
        #  - NLOS average power = p_first * (1/(K_lin+1))
        los_amp = np.sqrt(p_first * K_lin / (K_lin + 1))
        nlos_std = np.sqrt(p_first / (K_lin + 1) / 2)

        # LOS tap (tap 0)
        # The LOS component is a deterministic amplitude with random phase:
        phase_0 = np.random.uniform(0, 2*np.pi)
        los_comp = los_amp * np.exp(1j * phase_0)
        # The NLOS portion is Rayleigh:
        nlos_comp = nlos_std * (np.random.randn() + 1j*np.random.randn())
        chan_coef[0] = los_comp + nlos_comp

        for i in range(1, taps):
            rayleigh_std = np.sqrt(p_lin_norm[i] / 2)
            chan_coef[i] = rayleigh_std * (np.random.randn() + 1j*np.random.randn())
    else:
        for i in range(taps):
            rayleigh_std = np.sqrt(p_lin_norm[i] / 2)
            chan_coef[i] = rayleigh_std * (np.random.randn() + 1j*np.random.randn())

    max_speed_mps = max_speed * 1000 / 3600  # km/h -> m/s
    max_doppler_shift = (max_speed_mps * f_c) / 299792458.0  # in Hz
    doppler_res = 1 / (N * T)

    # version A:bulk Doppler shift for all taps (non-terrestrial channels)
    angle = np.random.uniform(0, 2*np.pi)
    bulk_doppler_shift = max_doppler_shift * np.cos(angle)
    doppler_shifts = np.full(taps, bulk_doppler_shift, dtype=float)
    doppler_taps = np.rint(doppler_shifts / doppler_res).astype(int)

    # version B:  random Doppler shifts for each tap (terrestrial channels)
    # max_Doppler_tap = max_doppler_shift / doppler_res
    # doppler_taps = max_Doppler_tap * np.cos(2 * np.pi * np.random.rand(taps)) # Doppler taps using Jake's spectrum
    # doppler_taps = np.rint(doppler_taps).astype(int)
    return chan_coef, delay_taps, doppler_taps, taps

@njit
def gen_discrete_time_channel(N, M, P, delay_taps, Doppler_taps, chan_coef):
    z = np.exp(1j * 2 * np.pi / (N * M))
    l_max = np.max(delay_taps)
    gs = np.zeros((l_max + 1, N * M), dtype=np.complex128)
    for q in range(N * M):
        for i in range(P):
            g_i = chan_coef[i]
            l_i = delay_taps[i]
            k_i = Doppler_taps[i]
            gs[l_i, q] += g_i * (z ** (k_i * (q - l_i)))
    return gs

@njit
def gen_delay_time_channel_vectors(N, M, l_max, gs):
    nu_ml_tilda = np.zeros((N, M, l_max + 1), dtype=np.complex128)
    for n in range(N):
        for m in range(M):
            for l in range(l_max):
                nu_ml_tilda[n, m, l] = gs[l, m + n * M]
    return nu_ml_tilda

@njit
def apply_channel(N, M, gs, s, L_set):
    r = np.zeros(N * M, dtype=np.complex128)
    for q in range(N * M):
        for i in range(len(L_set)):
            l = L_set[i]
            if q >= l:
                r[q] += gs[l, q] * s[q - l]
    return r

def generate_time_frequency_channel_zp(N, M, gs, L_set):
    H_t_f = np.zeros((N, M), dtype=np.complex128)
    Fm = dft_matrix(M)
    norm_Fm = np.linalg.norm(Fm, 2)
    Fm = Fm / norm_Fm

    Gn = np.zeros((M, M), dtype=np.complex128)
    for n in range(N):
        Gn.fill(0) 
        for m in range(M):
            for l in L_set:
                if m >= l:
                    Gn[m, m - l] = gs[l, m + n * M]
        product = Fm @ Gn @ Fm.conj().T
        H_t_f[n, :] = np.diag(product)
    return H_t_f

def generate_2d_data_grid(N, M, x_data, data_grid):
    x_vec = np.zeros(N * M, dtype=np.complex128)
    data_array = np.reshape(data_grid, (N * M,), order='F')
    data_pos = np.nonzero(data_array > 0)[0]
    x_data = np.ravel(x_data, order='F')
    x_vec[data_pos] = x_data
    X = np.reshape(x_vec, (M, N), order='F')
    return X

@njit
def compute_d_m_tilda(N, M, M_prime, L_set, nu_ml_tilda):
    d_m_tilda = np.zeros((N, M), dtype=np.complex128)
    for m in range(M_prime):
        for i in range(L_set.shape[0]):
            l = L_set[i]
            d_m_tilda[:, m] += np.abs(nu_ml_tilda[:, m + l, l])**2
    return d_m_tilda

@njit
def update_delta_y(N, M, L_set, nu_ml_tilda, x_m_tilda, delta_y_m_tilda):
    for m in range(M):
        for i in range(L_set.shape[0]):
            l = L_set[i]
            if m >= l:
                delta_y_m_tilda[:, m] -= nu_ml_tilda[:, m, l] * x_m_tilda[:, m - l]
    return delta_y_m_tilda

def mrc_delay_time_detector(N, M, M_data, M_mod, no, data_grid, r, H_tf, nu_ml_tilda, 
                              L_set, omega, decision, init_estimate, n_ite_MRC, mod_fn, demod_fn):
    Fn = dft_matrix(N)
    norm_Fn = np.linalg.norm(Fn, 2)
    Fn = Fn / norm_Fn

    N_syms_perfram = np.sum(data_grid > 0)
    data_array = np.reshape(data_grid, (N * M,), order='F')
    data_index = np.where(data_array > 0)[0]
    M_bits = np.log2(M_mod)
    N_bits_perfram = int(N_syms_perfram * M_bits)

    Y_tilda = np.reshape(r, (M, N), order='F')
    M_prime = M_data
    L_set = L_set

    if init_estimate == 1:
        Y_tf = np.fft.fft(Y_tilda, axis=0).T
        X_tf = np.conj(H_tf) * Y_tf / (H_tf * np.conj(H_tf) + no)
        X_est = np.fft.ifft(X_tf.T, axis=0) @ Fn

        indices = demod_fn(X_est, M_mod, output_type='int')
        X_est = mod_fn(indices, M_mod, input_type='int')
        X_est = np.reshape(X_est, (M, N), order='F')

        X_est = X_est * data_grid
        X_tilda_est = X_est @ Fn.conj().T
    else:
        X_est = np.zeros((M, N), dtype=np.complex128)
        X_tilda_est = X_est @ Fn.conj().T

    x_m = X_est.T
    x_m_tilda = X_tilda_est.T

    d_m_tilda = np.zeros((N, M), dtype=np.complex128)
    y_m_tilda = np.reshape(r, (M, N), order='F').T
    delta_y_m_tilda = y_m_tilda.copy()

    for m in range(M_prime):
            for l in L_set:
                d_m_tilda[:, m] += np.abs(nu_ml_tilda[:, m + l, l]) ** 2

    for m in range(M):
            for l in L_set:
                if m >= l:
                    delta_y_m_tilda[:, m] -= nu_ml_tilda[:, m, l] * x_m_tilda[:, m - l]

    # L_set_arr = np.array(L_set, dtype=np.int64)
    # d_m_tilda = compute_d_m_tilda(N, M, M_prime, L_set_arr, nu_ml_tilda)
    # delta_y_m_tilda = update_delta_y(N, M, L_set_arr, nu_ml_tilda, x_m_tilda, delta_y_m_tilda)

    x_m_tilda_old = x_m_tilda.copy()
    c_m_tilda = x_m_tilda.copy()

    error = np.zeros(n_ite_MRC)
    for ite in range(n_ite_MRC):
        delta_g_m_tilda = np.zeros((N, M), dtype=np.complex128)
        for m in range(M_prime):
            for l in L_set:
                delta_g_m_tilda[:, m] += np.conj(nu_ml_tilda[:, m + l, l]) * delta_y_m_tilda[:, m + l]
            c_m_tilda[:, m] = x_m_tilda_old[:, m] + delta_g_m_tilda[:, m] / d_m_tilda[:, m]
            if decision == 1:
                indices = demod_fn(Fn @ c_m_tilda[:, m], M_mod, output_type='int')
                x_m[:, m] = mod_fn(indices, M_mod, input_type='int')
                x_m_tilda[:, m] = (1 - omega) * c_m_tilda[:, m] + omega * (Fn.conj().T @ x_m[:, m])
            else:
                x_m_tilda[:, m] = c_m_tilda[:, m]
            for l in L_set:
                delta_y_m_tilda[:, m + l] -= nu_ml_tilda[:, m + l, l] * (x_m_tilda[:, m] - x_m_tilda_old[:, m])
            x_m_tilda_old[:, m] = x_m_tilda[:, m]

        error[ite] = np.linalg.norm(delta_y_m_tilda, 2)
        if ite > 0:
            if error[ite] >= error[ite - 1]:  # Line 15 of Algorithm 2 in [R1]
                break

    if n_ite_MRC == 0:
        ite = 0

    X_est = (Fn @ x_m_tilda).T
    x_est = np.reshape(X_est, (N * M,), order='F')
    x_data = x_est[data_index]

    est_bits = demod_fn(x_data, M_mod, output_type='bit')
    est_bits = np.reshape(est_bits, (N_bits_perfram, 1), order='F')
    return est_bits, ite, x_data

@njit
def dft_matrix(N):
    n = np.arange(N)
    k = np.arange(N)
    return np.exp(-2j * np.pi * np.outer(n, k) / N)

def block_LMMSE_detector(N, M, M_mod, noise_var, data_grid, r, gs, L_set, demod_fn):
    Fn = dft_matrix(N)
    norm_Fn = np.linalg.norm(Fn, 2)
    Fn = Fn / norm_Fn

    N_syms_perfram = np.sum(data_grid > 0)
    data_array = np.reshape(data_grid, (N * M,), order='F')
    data_index = np.where(data_array > 0)[0]
    M_bits = np.log2(M_mod)
    N_bits_perfram = int(N_syms_perfram * M_bits)

    sn_block_est = np.zeros((M, N), dtype=complex)

    for n in range(N):
        Gn = np.zeros((M, M), dtype=complex)
        for m in range(M):
            for l in (L_set + 1): 
                if (m + 1) >= l:
                    Gn[m, (m + 1) - int(l)] = gs[int(l) - 1, m + n * M]
        rn = r[n * M: (n + 1) * M]
        Rn = np.dot(np.conj(Gn).T, Gn)
        sn_block_est[:, n] = np.linalg.inv(Rn + noise_var * np.eye(M)) @ (np.dot(np.conj(Gn).T, rn))

    X_tilda_est = sn_block_est
    X_est = np.dot(X_tilda_est, Fn)
    x_est = np.reshape(X_est, (N * M,), order='F')
    x_data = x_est[data_index]

    est_bits = demod_fn(x_data, M_mod, output_type='bit')
    est_bits = est_bits.reshape((N_bits_perfram, 1))
    return est_bits, x_data
