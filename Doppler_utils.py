import numpy as np
from numba import njit

def dft_matrix(N):
    W = np.empty((N, N), dtype=np.complex128)
    for n in range(N):
        for k in range(N):
            W[n, k] = np.exp(-2j * np.pi * n * k / N)
    return W

def generate_delay_Doppler_channel_parameters(N, M, car_fre, delta_f, T, max_speed):
    one_delay_tap = 1 / (M * delta_f)
    one_doppler_tap = 1 / (N * T)
    delays = np.array([0, 30, 150, 310, 370, 710, 1090, 1730, 2510]) * 1e-9
    taps = len(delays)
    delay_taps = np.rint(delays / one_delay_tap).astype(np.int64)
    pdp = np.array([0, -1.5, -1.4, -3.6, -0.6, -9.1, -7.0, -12.0, -16.9])
    pow_prof = 10 ** (pdp / 10)
    pow_prof = pow_prof / np.sum(pow_prof)
    chan_coef = np.sqrt(pow_prof) * (np.sqrt(1/2) * (np.random.randn(taps) + 1j * np.random.randn(taps)))
    max_UE_speed = max_speed * (1000 / 3600)  # km/hr to m/s
    Doppler_vel = (max_UE_speed * car_fre) / 299792458
    max_Doppler_tap = Doppler_vel / one_doppler_tap
    Doppler_taps = max_Doppler_tap * np.cos(2 * np.pi * np.random.rand(taps))
    return chan_coef, delay_taps, Doppler_taps, taps

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
        Gn.fill(0)  # 각 시간 슬롯마다 초기화
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
            # nu_ml_tilda[:, m+l, l]의 절댓값 제곱을 누적
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

def mrc_low_complexity(N, M, M_mod, no, data_grid, r, H_tf, gs, L_set, omega, decision, init_estimate, n_ite, demod_fn, mod_fn):
    Fn = dft_matrix(N)          # dft_matrix should generate an N×N DFT matrix
    Fn = Fn / np.linalg.norm(Fn, 2)

    N_syms_perfram = np.sum(data_grid > 0)
    data_array = np.reshape(data_grid, (N * M,), order='F')
    data_index = np.where(data_array > 0)[0]   # indices of data symbols (0-indexed)
    data_location = np.reshape(data_grid, (N * M, 1), order='F')

    M_bits = int(np.log2(M_mod))
    N_bits_perfram = int(N_syms_perfram * M_bits)

    Y_tilda = np.reshape(r, (M, N), order='F')
    if init_estimate == 1:
        Y_tf = np.fft.fft(Y_tilda, axis=0).T
        X_tf = np.conj(H_tf) * Y_tf / (H_tf * np.conj(H_tf) + no)
        X_est = np.fft.ifft(X_tf.T, axis=0) @ Fn
        X_est = mod_fn(demod_fn(X_est, M_mod, output_type='int'), M_mod, input_type='int')
        X_est = np.reshape(X_est, (M, N), order='F')

        X_est = X_est * data_grid
        X_tilda_est = X_est @ Fn.conj().T
    else:
        X_est = np.zeros((M, N), dtype=complex)
        X_tilda_est = X_est @ Fn.conj().T
    X_tilda_est = X_tilda_est * data_grid

    error = np.zeros(n_ite)
    s_est = np.reshape(X_tilda_est, (N * M,), order='F').astype(complex)
    delta_r = r.copy().astype(complex)
    d = np.zeros(N * M, dtype=float)

    for q in range(N * M):
        if data_location[q, 0] == 1:
            for l in (L_set):  # l takes MATLAB-style indices (e.g., if L_set = [0,2] then l in [1,3])
                d[q] += np.abs(gs[int(l), q + int(l)])**2

    for q in range(N * M):
        for l in (L_set + 1):
            if (q + 1) >= l:  # converting to MATLAB 1-index
                delta_r[q] = delta_r[q] - gs[int(l) - 1, q] * s_est[q - int(l) + 1]

    for ite in range(n_ite):
        delta_g = np.zeros(N * M, dtype=complex)
        s_est_old = s_est.copy()
        for q in data_index:
            for l in L_set:
                delta_g[q] += np.conj(gs[l, q + l]) * delta_r[q + l]
            s_est[q] = s_est_old[q] + delta_g[q] / d[q]

            for l in L_set:
                delta_r[q + int(l)] = delta_r[q + int(l)] - \
                    gs[int(l), q + int(l)] * (s_est[q] - s_est_old[q])

        s_est_old = s_est.copy()

        if decision == 1:
            X_est = np.reshape(s_est, (M, N), order='F') @ Fn
            temp = mod_fn(demod_fn(X_est, M_mod, output_type='int'), M_mod, input_type='int')
            temp = np.reshape(temp, (M, N), order='F')
            X_tilda_est = (temp * data_grid) @ Fn.conjugate().T
            s_est = (1 - omega) * s_est + omega * np.reshape(X_tilda_est, (N * M,), order='F')

        for q in data_index:
            for l in (L_set):
                delta_r[q + int(l)] = delta_r[q + int(l)] - \
                    gs[int(l), q + int(l)] * (s_est[q] - s_est_old[q])

        curr_error = np.linalg.norm(delta_r, 2)
        error[ite] = curr_error
        if ite > 0:
            if error[ite] >= error[ite - 1]:
                ite = ite + 1  # count current iteration
                break
    else:
        ite = n_ite  # if loop completes without break

    if n_ite == 0:
        ite = 0

    X_tilda_est = np.reshape(s_est, (M, N), order='F')
    X_est = X_tilda_est @ Fn
    x_est = np.reshape(X_est, (N * M,), order='F')
    x_data = x_est[data_index]
    est_bits = demod_fn(x_data, M_mod, output_type='bit')
    est_bits = np.reshape(est_bits, (N_bits_perfram, 1), order='F')
    return est_bits, ite, x_data

def TF_single_tap_equalizer(N, M, M_mod, noise_var, data_grid, Y, H_tf, demod_fn):
    Fn = dft_matrix(N)
    Fn = Fn / np.linalg.norm(Fn, 2)

    N_syms_perfram = np.sum(data_grid > 0)
    data_array = np.reshape(data_grid, (N * M,), order='F')
    data_index = np.where(data_array > 0)[0]

    M_bits = int(np.log2(M_mod))
    N_bits_perfram = N_syms_perfram * M_bits

    Y_tf = np.fft.fft(Y @ Fn.conj().T, axis=0).T
    X_tf = np.conj(H_tf) * Y_tf / (H_tf * np.conj(H_tf) + noise_var)
    X_est = np.fft.ifft(X_tf.T, axis=0) @ Fn

    x_est = np.reshape(X_est, (N * M,), order='F')
    x_data = x_est[data_index]
    est_bits = demod_fn(x_data, M_mod, output_type='bit')
    est_bits = np.reshape(est_bits, (N_bits_perfram, 1), order='F')
    return est_bits, x_data

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
