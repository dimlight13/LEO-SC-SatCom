import numpy as np

def perform_modulate(code_bits, M_mod, modulate_fn):
    symbols = modulate_fn(code_bits.numpy(), M_mod)
    return symbols

def perform_demodulate(symbols, M_mod, demodulate_fn):
    bits = demodulate_fn(symbols.numpy(), M_mod)
    return bits

def perform_soft_demodulate(symbols, M_mod, demodulate_fn, demod_type='soft', noise_var=1.0):
    bits = demodulate_fn(symbols.numpy(), M_mod, demod_type=demod_type, output_type='bit', noisevar=noise_var)
    return bits

def logsumexp(a, axis=None, keepdims=False):
    a_max = np.max(a, axis=axis, keepdims=True)
    sum_exp = np.sum(np.exp(a - a_max), axis=axis, keepdims=keepdims)
    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)
    return np.log(sum_exp) + a_max

def modulate_psk(code_bits, M_mod, input_type='bit'):
    code_bits = np.asarray(code_bits)

    if M_mod == 2:
        if input_type == 'bit':
            code_bits = code_bits.astype(np.float32)
            if not np.all(np.isin(code_bits, [0, 1])):
                raise ValueError("BPSK input must be binary (0 or 1) for input_type='bit'")
            symbols = 2 * code_bits - 1
            return symbols.astype(np.complex64)
        elif input_type == 'int':
            if not np.all(np.isin(code_bits, [0, 1])):
                raise ValueError("BPSK input must be in {0, 1} for input_type='int'")
            mapping = np.array([-1, 1], dtype=np.complex64)
            return mapping[code_bits]
        else:
            raise ValueError("Invalid input_type for PSK modulation.")
    elif M_mod == 4:
        scale = np.sqrt(2.0)  
        if input_type == 'bit':
            code_bits = code_bits.astype(np.float32)
            if not np.all(np.isin(code_bits, [0, 1])):
                raise ValueError("QPSK input must be binary (0 or 1) for input_type='bit'")
            code_bits_reshaped = code_bits.reshape(-1, 2, order='C')
            i = 2 * code_bits_reshaped[:, 0] - 1  
            q = 2 * code_bits_reshaped[:, 1] - 1  
            symbols = i / scale + 1j * q / scale  
            return symbols.astype(np.complex64)
        elif input_type == 'int':
            if not np.all(np.isin(code_bits, [0, 1, 2, 3])):
                raise ValueError("QPSK input must be in {0, 1, 2, 3} for input_type='int'")
            mapping = np.array([-1-1j, -1+1j, 1-1j, 1+1j], dtype=np.complex64) / scale
            return mapping[code_bits]
        else:
            raise ValueError("Invalid input_type for PSK modulation.")
    else:
        raise ValueError("Unsupported M_mod for PSK modulation (only 2 and 4 supported).")

def demodulate_psk(symbols, M_mod, demod_type='hard', output_type='bit', noisevar=None):
    symbols = np.asarray(symbols)
    if M_mod not in [2, 4]:
        raise ValueError("PSK demodulation only supports M_mod=2 (BPSK) or 4 (QPSK).")
    if demod_type == 'hard':
        if M_mod == 2:
            bits = (symbols.real > 0).astype(np.float32)
            if output_type == 'bit':
                return bits
            elif output_type == 'int':
                return bits.astype(np.int32)
            else:
                raise ValueError("For PSK hard demodulation, the output_type must be 'bit' or 'int'.")
        elif M_mod == 4:
            real_part = symbols.real
            imag_part = symbols.imag
            bits_i = (real_part > 0).astype(np.float32)
            bits_q = (imag_part > 0).astype(np.float32)
            if output_type == 'bit':
                return np.stack((bits_i, bits_q), axis=-1).flatten()
            elif output_type == 'int':
                int_values = 2 * bits_i + bits_q
                return int_values.astype(np.int32)
            else:
                raise ValueError("For PSK hard demodulation, the output_type must be 'bit' or 'int'.")
    elif demod_type == 'soft':
        if noisevar is None:
            raise ValueError("For soft demodulation, noisevar is required (noise variance).")
        if M_mod == 2:
            llr = (2.0 * symbols.real) / noisevar
            return llr
        elif M_mod == 4:
            real = symbols.real
            imag = symbols.imag
            llr_i = (2 * np.sqrt(2) * real) / noisevar
            llr_q = (2 * np.sqrt(2) * imag) / noisevar
            return np.column_stack((llr_i, llr_q)).reshape(-1)
    else:
        raise ValueError("demod_type must be either 'hard' or 'soft'.")

def binary_to_gray(bits):
    num_bits = bits.shape[1]
    exponents = np.arange(num_bits - 1, -1, -1)
    powers_of_two = np.left_shift(1, exponents)  # 2^exponents
    bits_int = np.sum(bits * powers_of_two, axis=1)
    gray_int = np.bitwise_xor(bits_int, np.right_shift(bits_int, 1))
    gray_bits = (np.right_shift(gray_int[:, None], exponents) & 1).astype(int)
    return gray_bits

def gray_to_binary(gray):
    gray = np.asarray(gray, dtype=int)
    binary = np.zeros_like(gray, dtype=int)
    binary[:, 0] = gray[:, 0]
    for i in range(1, gray.shape[1]):
        binary[:, i] = binary[:, i - 1] ^ gray[:, i]
    return binary

def generate_qam_lut(M):
    n_bits_per_axis = int(np.log(M) / np.log(4))  # log4(M)

    num_codes = 2 ** n_bits_per_axis
    gray_codes = [i ^ (i >> 1) for i in range(num_codes)]

    num_levels = 2 ** n_bits_per_axis  # Number of levels per axis (e.g., 4 for 16-QAM)
    symbol_levels = np.array([2 * i - num_levels + 1 for i in range(num_levels)])  # Compute symbol levels
    gray_to_level = {gc: level for gc, level in zip(gray_codes, symbol_levels)}  # Map Gray code to levels
    lut = np.array([[gray_to_level[gray_codes[i]], gray_to_level[gray_codes[q]]]
                    for i in range(num_levels) for q in range(num_levels)], dtype=np.float32)
    return lut

def modulate_qam(code_bits, M_mod, input_type='bit'):
    lut = generate_qam_lut(M_mod)
    scale = np.sqrt((2.0 / 3.0) * (M_mod - 1.0))
    n_bits_axis = int(np.log(M_mod) / np.log(4))  # log4(M)

    if input_type == 'bit':
        code_bits = np.array(code_bits).reshape(-1, 2 * n_bits_axis)
        bits_I = code_bits[:, :n_bits_axis]
        bits_Q = code_bits[:, n_bits_axis:]
        binary_I = gray_to_binary(bits_I).astype(np.int32)
        binary_Q = gray_to_binary(bits_Q).astype(np.int32)
        weights = 2 ** np.arange(n_bits_axis - 1, -1, -1)
        I_index = np.dot(binary_I, weights)
        Q_index = np.dot(binary_Q, weights)
        decimal_index = I_index * (2 ** n_bits_axis) + Q_index
    elif input_type == 'int':
        decimal_index = np.array(code_bits).astype(np.int32).flatten()
    else:
        raise ValueError("Invalid input_type: {}".format(input_type))
    iq_pairs = lut[decimal_index]
    real_part = iq_pairs[:, 0] / scale
    imag_part = iq_pairs[:, 1] / scale
    symbols = real_part + 1j * imag_part
    return symbols.astype(np.complex64)

def generate_qam_bit_mapping(n_bits_axis):
    n_symbols = 2 ** (2 * n_bits_axis)
    mapping = np.zeros((n_symbols, 2 * n_bits_axis), dtype=np.int32)
    for idx in range(n_symbols):
        I_index = idx // (2 ** n_bits_axis)
        Q_index = idx % (2 ** n_bits_axis)
        def int_to_bin(x, n):
            return np.array([(x >> i) & 1 for i in range(n - 1, -1, -1)], dtype=np.int32)
        bin_I = int_to_bin(I_index, n_bits_axis)
        bin_Q = int_to_bin(Q_index, n_bits_axis)
        gray_I = binary_to_gray(bin_I.reshape(1, -1))[0]
        gray_Q = binary_to_gray(bin_Q.reshape(1, -1))[0]
        mapping[idx, :] = np.concatenate((gray_I, gray_Q))
    return mapping.astype(np.float32)

def demodulate_qam(symbols, M_mod, demod_type='hard', output_type='bit', noisevar=None):
    symbols = np.asarray(symbols)
    n_bits_axis = int(np.log(M_mod) / np.log(4))
    scale = np.sqrt((2.0 / 3.0) * (M_mod - 1.0))

    if demod_type == 'hard':
        lut = generate_qam_lut(M_mod)
        symbols_flat = symbols.flatten()
        real_part = symbols_flat.real * scale
        imag_part = symbols_flat.imag * scale
        lut_real = lut[:, 0]
        lut_imag = lut[:, 1]
        distances = (real_part[:, None] - lut_real[None, :])**2 + (imag_part[:, None] - lut_imag[None, :])**2
        decimal_index = np.argmin(distances, axis=1)
        if output_type == 'int':
            return decimal_index.astype(np.int32)
        elif output_type == 'bit':
            I_index = decimal_index // (2 ** n_bits_axis)
            Q_index = decimal_index % (2 ** n_bits_axis)
            def int_to_binary_array(x, n):
                return ((x[:, None] & (1 << np.arange(n - 1, -1, -1))) > 0).astype(np.int32)
            binary_I = int_to_binary_array(I_index, n_bits_axis)
            binary_Q = int_to_binary_array(Q_index, n_bits_axis)
            gray_I = binary_to_gray(binary_I)
            gray_Q = binary_to_gray(binary_Q)
            gray_bits = np.hstack((gray_I, gray_Q))
            return gray_bits.flatten(order='C').astype(np.float32)
        else:
            raise ValueError("For QAM hard demodulation, the output_type must be 'bit' or 'int'.")
    elif demod_type == 'soft':
        if noisevar is None:
            raise ValueError("When soft demodulating, noisevar is required (noise variance).")
        bitsPerSymbol = 2 * n_bits_axis
        lut = generate_qam_lut(M_mod)
        symSet = (lut[:, 0] + 1j * lut[:, 1]) / scale
        bitMap = generate_qam_bit_mapping(n_bits_axis)
        distances = np.abs(symbols[:, None] - symSet[None, :]) ** 2

        demodLLR = []
        for b in range(bitsPerSymbol):
            idx1 = np.where(bitMap[:, b] == 1)[0]
            idx0 = np.where(bitMap[:, b] == 0)[0]
            LLR = logsumexp(-distances[:, idx1] / noisevar, axis=1) - logsumexp(-distances[:, idx0] / noisevar, axis=1)
            demodLLR.append(LLR)
        demodLLR = np.stack(demodLLR, axis=1)
        return demodLLR.reshape(-1)
    
    else:
        raise ValueError("demod_type must be 'hard' or 'soft'.")
