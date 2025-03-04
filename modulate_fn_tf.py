import tensorflow as tf

def generate_qam_lut_tf(M):
    n_bits_per_axis = int(tf.math.log(tf.cast(M, tf.float32)) / tf.math.log(4.0))
    num_codes = 2 ** n_bits_per_axis  # 각 축의 심볼 수
    symbol_levels = tf.constant([2 * i - num_codes + 1 for i in range(num_codes)], dtype=tf.float32)
    i_vals = symbol_levels  # shape [num_codes]
    i_mat = tf.reshape(tf.tile(i_vals, [num_codes]), [num_codes, num_codes])
    q_mat = tf.transpose(i_mat)
    lut = tf.stack([tf.reshape(i_mat, [-1]), tf.reshape(q_mat, [-1])], axis=1)
    return lut

lut_16 = generate_qam_lut_tf(16)
lut_64 = generate_qam_lut_tf(64)
lut_256 = generate_qam_lut_tf(256)

_EXPONENTS_4 = tf.constant([3, 2, 1, 0], dtype=tf.int32)
_EXPONENTS_6 = tf.constant([5, 4, 3, 2, 1, 0], dtype=tf.int32)
_EXPONENTS_8 = tf.constant([7, 6, 5, 4, 3, 2, 1, 0], dtype=tf.int32)

def binary_to_gray(bits):
    bits = tf.cast(bits, tf.int32)
    num_bits = tf.shape(bits)[1]

    if num_bits == 4:
        exponents = _EXPONENTS_4
    elif num_bits == 6:
        exponents = _EXPONENTS_6
    elif num_bits == 8:
        exponents = _EXPONENTS_8
    else:
        exponents = tf.range(num_bits - 1, -1, -1, dtype=tf.int32)

    powers_of_two = tf.bitwise.left_shift(1, exponents)
    bits_int = tf.reduce_sum(bits * powers_of_two, axis=1)

    gray_int = tf.bitwise.bitwise_xor(bits_int, tf.bitwise.right_shift(bits_int, 1))
    gray_bits = tf.bitwise.right_shift(
        tf.expand_dims(gray_int, -1), exponents) & 1
    return gray_bits

def gray_to_binary(gray):
    gray = tf.cast(gray, tf.int32)
    binary = tf.scan(
        lambda acc, x: acc ^ x,
        tf.transpose(gray),
        initializer=tf.zeros(tf.shape(gray)[0], dtype=tf.int32)
    )
    return tf.transpose(binary)

def modulate_psk(code_bits, M_mod):
    if M_mod == 2:
        code_bits = tf.cast(code_bits, tf.float32)
        symbols_real = 2 * code_bits - 1
        return tf.complex(symbols_real, tf.zeros_like(symbols_real))

    elif M_mod == 4:
        scale = tf.sqrt(2.0)  # 정규화 상수: 1/sqrt(2)
        code_bits = tf.cast(code_bits, tf.float32)
        code_bits_reshaped = tf.reshape(code_bits, [-1, 2])
        i = 2 * code_bits_reshaped[:, 0] - 1
        q = 2 * code_bits_reshaped[:, 1] - 1
        return tf.complex(i / scale, q / scale)
    else:
        raise ValueError("Unsupported M_mod for PSK modulation (only 2 and 4 supported).")

def demodulate_psk(symbols, M_mod):
    if M_mod not in [2, 4]:
        raise ValueError("PSK demodulation only supports M_mod=2 (BPSK) or 4 (QPSK).")
    if M_mod == 2:
        bits = tf.cast(tf.math.greater(tf.math.real(symbols), 0), tf.float32)
        return bits

    elif M_mod == 4:
        real_part = tf.math.real(symbols)
        imag_part = tf.math.imag(symbols)
        bits_i = tf.cast(tf.math.greater(real_part, 0), tf.float32)
        bits_q = tf.cast(tf.math.greater(imag_part, 0), tf.float32)
        bits = tf.reshape(tf.stack([bits_i, bits_q], axis=1), [-1])
        return bits

def modulate_qam(code_bits, M_mod):
    if M_mod == 16:
        lut = lut_16
    elif M_mod == 64:
        lut = lut_64
    elif M_mod == 256:
        lut = lut_256

    scale = tf.sqrt((2.0 / 3.0) * tf.cast(M_mod - 1, tf.float32))
    n_bits_axis = int(tf.math.log(tf.cast(M_mod, tf.float32)) / tf.math.log(4.0))

    code_bits = tf.reshape(tf.cast(code_bits, tf.int32), [-1, 2 * n_bits_axis])
    bits_I = code_bits[:, :n_bits_axis]
    bits_Q = code_bits[:, n_bits_axis:]

    bits_I = gray_to_binary(bits_I)
    bits_Q = gray_to_binary(bits_Q)

    exponents = tf.range(n_bits_axis - 1, -1, -1, dtype=tf.int32)
    weights = tf.pow(2, exponents)  # shape [n_bits_axis]
    weights = tf.reshape(weights, [n_bits_axis, 1])  # shape [n_bits_axis, 1]
    I_index = tf.matmul(tf.cast(bits_I, tf.int32), weights)  # shape [N, 1]
    Q_index = tf.matmul(tf.cast(bits_Q, tf.int32), weights)  # shape [N, 1]
    I_index = tf.reshape(I_index, [-1])
    Q_index = tf.reshape(Q_index, [-1])
    decimal_index = I_index * (2 ** n_bits_axis) + Q_index

    iq_pairs = tf.gather(lut, decimal_index)  # shape [num_symbols, 2]
    real_part = iq_pairs[:, 0] / scale
    imag_part = iq_pairs[:, 1] / scale
    symbols = tf.complex(real_part, imag_part)
    return symbols

def demodulate_qam(symbols, M_mod):
    symbols = tf.convert_to_tensor(symbols)
    n_bits_axis = int(tf.math.log(tf.cast(M_mod, tf.float32)) / tf.math.log(4.0))
    scale = tf.sqrt((2.0 / 3.0) * tf.cast(M_mod - 1, tf.float32))

    if M_mod == 16:
        lut_tf = lut_16
    elif M_mod == 64:
        lut_tf = lut_64
    elif M_mod == 256:
        lut_tf = lut_256

    symbols_flat = tf.reshape(symbols, [-1])
    real_part = tf.math.real(symbols_flat) * scale
    imag_part = tf.math.imag(symbols_flat) * scale
    lut_real = lut_tf[:, 0]
    lut_imag = lut_tf[:, 1]
    distances = (tf.expand_dims(real_part, 1) - tf.expand_dims(lut_real, 0))**2 + \
                (tf.expand_dims(imag_part, 1) - tf.expand_dims(lut_imag, 0))**2
    decimal_index = tf.argmin(distances, axis=1, output_type=tf.int32)
    I_index = decimal_index // (2 ** n_bits_axis)
    Q_index = decimal_index % (2 ** n_bits_axis)
    def int_to_binary_array(x, n):
        exps = tf.range(n - 1, -1, -1, dtype=tf.int32)
        x_expanded = tf.expand_dims(x, 1)
        bits = tf.bitwise.bitwise_and(tf.bitwise.right_shift(x_expanded, exps), 1)
        return bits  # shape [N, n]
    binary_I = int_to_binary_array(I_index, n_bits_axis)
    binary_Q = int_to_binary_array(Q_index, n_bits_axis)

    gray_I = binary_to_gray(binary_I)
    gray_Q = binary_to_gray(binary_Q)

    bits = tf.concat([gray_I, gray_Q], axis=1)
    bits = tf.reshape(bits, [-1])
    return tf.cast(bits, tf.float32)
