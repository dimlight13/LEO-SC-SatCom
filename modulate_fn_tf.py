import tensorflow as tf

def generate_qam_lut_tf(M):
    n_bits_per_axis = int(tf.math.log(tf.cast(M, tf.float32)) / tf.math.log(4.0))
    num_codes = 2 ** n_bits_per_axis
    symbol_levels = tf.constant([2 * i - num_codes + 1 for i in range(num_codes)], dtype=tf.float32)
    i_vals = symbol_levels
    i_mat = tf.reshape(tf.tile(i_vals, [num_codes]), [num_codes, num_codes])
    q_mat = tf.transpose(i_mat)
    lut = tf.stack([tf.reshape(i_mat, [-1]), tf.reshape(q_mat, [-1])], axis=1)
    return lut

lut_16 = generate_qam_lut_tf(16)
lut_64 = generate_qam_lut_tf(64)
lut_256 = generate_qam_lut_tf(256)

@tf.function
def binary_to_gray(bits):
    bits = tf.cast(bits, tf.int32)
    static_shape = bits.shape
    if static_shape[1] is not None:
        num_bits = static_shape[1]
        if num_bits == 2:
            exponents = tf.constant([1, 0], dtype=tf.int32)
        elif num_bits == 3:
            exponents = tf.constant([2, 1, 0], dtype=tf.int32)
        elif num_bits == 4:
            exponents = tf.constant([3, 2, 1, 0], dtype=tf.int32)
        else:
            exponents = tf.range(num_bits - 1, -1, -1, dtype=tf.int32)
    else:
        num_bits = tf.shape(bits)[1]
        exponents = tf.switch_case(
            num_bits - 2,  # Adjust to make indices contiguous starting from 0
            branch_fns={
                0: lambda: tf.constant([1, 0], dtype=tf.int32),
                1: lambda: tf.constant([2, 1, 0], dtype=tf.int32),
                2: lambda: tf.constant([3, 2, 1, 0], dtype=tf.int32)
            },
            default=lambda: tf.range(num_bits - 1, -1, -1, dtype=tf.int32)
        )
    powers_of_two = tf.bitwise.left_shift(1, exponents)
    bits_int = tf.reduce_sum(bits * powers_of_two, axis=1)
    gray_int = tf.bitwise.bitwise_xor(bits_int, tf.bitwise.right_shift(bits_int, 1))
    gray_bits = tf.bitwise.right_shift(tf.expand_dims(gray_int, -1), exponents) & 1
    return gray_bits

def gray_to_binary(gray):
    gray = tf.cast(gray, tf.int32)
    binary = tf.math.floormod(tf.math.cumsum(gray, axis=1), 2)
    return binary

@tf.function
def modulate_psk(code_bits, M_mod):
    M_mod_static = tf.get_static_value(M_mod)
    if M_mod_static is not None:
        M_mod_int = int(M_mod_static)
        if M_mod_int == 2:
            code_bits_cast = tf.cast(code_bits, tf.float32)
            symbols_real = 2 * code_bits_cast - 1
            return tf.complex(symbols_real, tf.zeros_like(symbols_real))
        elif M_mod_int == 4:
            scale = tf.sqrt(2.0)
            code_bits_cast = tf.cast(code_bits, tf.float32)
            code_bits_reshaped = tf.reshape(code_bits_cast, [-1, 2])
            i = 2 * code_bits_reshaped[:, 0] - 1
            q = 2 * code_bits_reshaped[:, 1] - 1
            return tf.complex(i / scale, q / scale)
        else:
            return tf.complex(tf.zeros_like(tf.cast(code_bits, tf.float32)),
                              tf.zeros_like(tf.cast(code_bits, tf.float32)))
    else:
        return tf.cond(
            tf.equal(M_mod, 2),
            lambda: (lambda: tf.complex(2 * tf.cast(code_bits, tf.float32) - 1, 
                                          tf.zeros_like(tf.cast(code_bits, tf.float32))))(),
            lambda: tf.cond(
                tf.equal(M_mod, 4),
                lambda: (lambda: tf.complex(
                    (2 * tf.reshape(tf.cast(code_bits, tf.float32), [-1, 2])[:,0] - 1) / tf.sqrt(2.0),
                    (2 * tf.reshape(tf.cast(code_bits, tf.float32), [-1, 2])[:,1] - 1) / tf.sqrt(2.0)
                ))(),
                lambda: tf.complex(tf.zeros_like(tf.cast(code_bits, tf.float32)), 
                                   tf.zeros_like(tf.cast(code_bits, tf.float32)))
            )
        )

@tf.function
def demodulate_psk(symbols, M_mod):
    M_mod_static = tf.get_static_value(M_mod)
    if M_mod_static is not None:
        M_mod_int = int(M_mod_static)
        if M_mod_int == 2:
            bits = tf.cast(tf.math.greater(tf.math.real(symbols), 0), tf.int32)
            return bits
        elif M_mod_int == 4:
            real_part = tf.math.real(symbols)
            imag_part = tf.math.imag(symbols)
            bits_i = tf.cast(tf.math.greater(real_part, 0), tf.int32)
            bits_q = tf.cast(tf.math.greater(imag_part, 0), tf.int32)
            return tf.reshape(tf.stack([bits_i, bits_q], axis=1), [-1])
        else:
            return tf.zeros_like(tf.math.real(symbols), dtype=tf.int32)
    else:
        return tf.cond(
            tf.equal(M_mod, 2),
            lambda: tf.cast(tf.math.greater(tf.math.real(symbols), 0), tf.int32),
            lambda: tf.cond(
                tf.equal(M_mod, 4),
                lambda: tf.reshape(tf.stack([
                        tf.cast(tf.math.greater(tf.math.real(symbols), 0), tf.int32),
                        tf.cast(tf.math.greater(tf.math.imag(symbols), 0), tf.int32)
                    ], axis=1), [-1]),
                lambda: tf.zeros_like(tf.math.real(symbols), dtype=tf.int32)
            )
        )

@tf.function
def modulate_qam(code_bits, M_mod):
    M_mod_static = tf.get_static_value(M_mod)
    if M_mod_static is None:
        return tf.cond(
            tf.equal(M_mod, 16),
            lambda: modulate_qam_common(code_bits, tf.sqrt((2.0/3.0)*15.0), 2, lut_16),
            lambda: tf.cond(
                tf.equal(M_mod, 64),
                lambda: modulate_qam_common(code_bits, tf.sqrt((2.0/3.0)*63.0), 3, lut_64),
                lambda: tf.cond(
                    tf.equal(M_mod, 256),
                    lambda: modulate_qam_common(code_bits, tf.sqrt((2.0/3.0)*255.0), 4, lut_256),
                    lambda: tf.complex(tf.zeros_like(tf.cast(code_bits, tf.float32)), 
                                       tf.zeros_like(tf.cast(code_bits, tf.float32)))
                )
            )
        )
    else:
        M_mod_int = int(M_mod_static)
        if M_mod_int == 16:
            return modulate_qam_common(code_bits, tf.sqrt((2.0/3.0)*15.0), 2, lut_16)
        elif M_mod_int == 64:
            return modulate_qam_common(code_bits, tf.sqrt((2.0/3.0)*63.0), 3, lut_64)
        elif M_mod_int == 256:
            return modulate_qam_common(code_bits, tf.sqrt((2.0/3.0)*255.0), 4, lut_256)
        else:
            return tf.complex(tf.zeros_like(tf.cast(code_bits, tf.float32)), 
                              tf.zeros_like(tf.cast(code_bits, tf.float32)))

def modulate_qam_common(code_bits, scale, n_bits_axis, lut):
    code_bits_reshaped = tf.reshape(tf.cast(code_bits, tf.int32), [-1, 2 * n_bits_axis])
    bits_I = code_bits_reshaped[:, :n_bits_axis]
    bits_Q = code_bits_reshaped[:, n_bits_axis:]
    bits_I = gray_to_binary(bits_I)
    bits_Q = gray_to_binary(bits_Q)
    exponents = tf.range(n_bits_axis - 1, -1, -1, dtype=tf.int32)
    weights = tf.pow(2, exponents)
    weights = tf.reshape(weights, [n_bits_axis, 1])
    I_index = tf.matmul(tf.cast(bits_I, tf.int32), weights)
    Q_index = tf.matmul(tf.cast(bits_Q, tf.int32), weights)
    I_index = tf.reshape(I_index, [-1])
    Q_index = tf.reshape(Q_index, [-1])
    decimal_index = I_index * (2 ** n_bits_axis) + Q_index
    iq_pairs = tf.gather(lut, decimal_index)
    real_part = iq_pairs[:, 0] / scale
    imag_part = iq_pairs[:, 1] / scale
    return tf.complex(real_part, imag_part)

@tf.function
def demodulate_qam(symbols, M_mod):
    M_mod_static = tf.get_static_value(M_mod)
    if M_mod_static is None:
        return tf.cond(
            tf.equal(M_mod, 16),
            lambda: demodulate_qam_common(symbols, tf.sqrt((2.0/3.0)*15.0), 2, lut_16),
            lambda: tf.cond(
                tf.equal(M_mod, 64),
                lambda: demodulate_qam_common(symbols, tf.sqrt((2.0/3.0)*63.0), 3, lut_64),
                lambda: tf.cond(
                    tf.equal(M_mod, 256),
                    lambda: demodulate_qam_common(symbols, tf.sqrt((2.0/3.0)*255.0), 4, lut_256),
                    lambda: tf.zeros_like(tf.math.real(symbols), dtype=tf.int32)
                )
            )
        )
    else:
        M_mod_int = int(M_mod_static)
        if M_mod_int == 16:
            return demodulate_qam_common(symbols, tf.sqrt((2.0/3.0)*15.0), 2, lut_16)
        elif M_mod_int == 64:
            return demodulate_qam_common(symbols, tf.sqrt((2.0/3.0)*63.0), 3, lut_64)
        elif M_mod_int == 256:
            return demodulate_qam_common(symbols, tf.sqrt((2.0/3.0)*255.0), 4, lut_256)
        else:
            return tf.zeros_like(tf.math.real(symbols), dtype=tf.int32)

def demodulate_qam_common(symbols, scale, n_bits_axis, lut_tf):
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
        return bits
    binary_I = int_to_binary_array(I_index, n_bits_axis)
    binary_Q = int_to_binary_array(Q_index, n_bits_axis)
    gray_I = binary_to_gray(binary_I)
    gray_Q = binary_to_gray(binary_Q)
    bits = tf.concat([gray_I, gray_Q], axis=1)
    return tf.reshape(bits, [-1])
