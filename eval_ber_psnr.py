"""
A single script combining the functionality from eval_ber.py and eval_psnr.py.
Use --eval_type ber or --eval_type psnr to switch between BER and PSNR evaluation.

Example usage:
python eval_ber_psnr.py --eval_type ber --snr_min -10 --snr_max 30
python eval_ber_psnr.py --eval_type psnr --snr_min 0 --snr_max 50
"""

import argparse
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import preprocess_data, load_models_from_dir

from models import VQVAE
from TurboCode import TurboEncoder, TurboDecoder
from modulate_fn_np import demodulate_psk, modulate_psk
from modulate_fn_np import modulate_qam, demodulate_qam
from modulate_fn_np import perform_modulate, perform_demodulate, perform_soft_demodulate
from utils import channel_effects, channel_effects_np
from Doppler_utils import dft_matrix, generate_2d_data_grid, generate_delay_Doppler_channel_parameters
from Doppler_utils import mrc_delay_time_detector, mrc_low_complexity, block_LMMSE_detector, TF_single_tap_equalizer
from Doppler_utils import gen_discrete_time_channel, gen_delay_time_channel_vectors, generate_time_frequency_channel_zp

MODULATION_ORDERS = [2, 4, 16, 64, 256]

def determine_modulation_index(args, EbNo, snr_boundaries):
    if args.modulation == 'auto':
        modulation_index = None
        for k in range(len(snr_boundaries) - 1):
            if snr_boundaries[k] <= EbNo < snr_boundaries[k + 1]:
                modulation_index = k
                break
        if modulation_index is None and EbNo < snr_boundaries[0]:
            modulation_index = 0
        if EbNo >= snr_boundaries[-1]:
            modulation_index = len(snr_boundaries) - 2
    elif args.modulation == "BPSK":
        modulation_index = 0
    elif args.modulation == "QPSK":
        modulation_index = 1
    elif args.modulation == "16QAM":
        modulation_index = 2
    elif args.modulation == "64QAM":
        modulation_index = 3
    elif args.modulation == "256QAM":
        modulation_index = 4
    return modulation_index

def calculate_snr(EbNo, modulation_order, is_channel_coding):
    EbNo_tf = tf.cast(EbNo, dtype=tf.float32)
    rate = 0.3328 if is_channel_coding else 1.0
    bits_per_symbol = np.log2(modulation_order)
    EsNo = EbNo_tf + 10 * np.log10(bits_per_symbol)
    snr_value = EsNo + 10 * np.log10(rate)
    return snr_value

def handle_channel_coding(encoded_bits, modulation_order, model, modulate_fn):
    bits_per_symbol = int(np.log2(modulation_order))
    bits_len = encoded_bits.shape[0] * encoded_bits.shape[1]
    remainder = (bits_len // encoded_bits.shape[0]) % bits_per_symbol

    M_mod = modulation_order
    if remainder == 0:
        symbols = perform_modulate(encoded_bits, M_mod, modulate_fn)
        symbols = tf.cast(symbols, dtype=tf.complex64)
        padding_bits = 0
    else:
        padding_bits = bits_per_symbol - remainder
        padded_code_bits = tf.pad(encoded_bits, [[0, 0], [0, padding_bits]], mode='CONSTANT', constant_values=0)
        symbols = perform_modulate(padded_code_bits, M_mod, modulate_fn)
        symbols = tf.cast(symbols, dtype=tf.complex64)

    return padding_bits, symbols

def handle_none_mode(frame, snr_eval, channel_type):
    frame = tf.reshape(frame, [1, -1])
    snr_eval = snr_eval[0]
    noisy_symbols = channel_effects(frame, snr_eval, channel_type)
    return noisy_symbols

def apply_doppler_channel(frame, snr_eval, channel_type, modulate_fn, demodulate_fn, args):
    # Map the modulation index to M_mod and omega.
    if args.modulation_index == 0:
        M_mod = 2
        omega = 1
    elif args.modulation_index == 1:
        M_mod = 4
        omega = 1
    elif args.modulation_index == 2:
        M_mod = 16
        omega = 1
    elif args.modulation_index == 3:
        M_mod = 64
        omega = 0.25
    elif args.modulation_index == 4:
        M_mod = 256
        omega = 0.25
    else:
        raise ValueError("Invalid modulation type")

    M = args.M_number
    N = args.N_number

    length_ZP = M / 16
    M_data = int(M - length_ZP)
    data_grid = np.zeros((M, N), dtype=np.float32)
    data_grid[:M_data, :] = 1

    snr_dB = snr_eval[0].numpy()
    car_fre = 20e9
    delta_f = 15e3
    T = 1 / delta_f

    Fn = dft_matrix(N)
    norm_Fn = np.linalg.norm(Fn, 2)
    Fn = Fn / norm_Fn

    eng_sqrt = 1 if M_mod == 2 else np.sqrt((M_mod - 1) / 6 * 4)
    SNR_linear = 10 ** (snr_dB / 10)
    sigma_2 = (np.abs(eng_sqrt) ** 2) / SNR_linear

    data = frame.numpy()
    X = generate_2d_data_grid(N, M, data, data_grid)

    X_tilda = X @ Fn.conj().T
    s = X_tilda.reshape(N * M, order='F')

    max_speed = 500  # km/hr
    chan_coef, delay_taps, Doppler_taps, taps = generate_delay_Doppler_channel_parameters(
        N, M, car_fre, delta_f, T, max_speed)
    L_set = np.unique(delay_taps)

    gs = gen_discrete_time_channel(N, M, taps, delay_taps, Doppler_taps, chan_coef)

    r = np.zeros(N * M, dtype=complex)
    l_max = int(max(delay_taps))

    for q in range(N * M):
        for l in L_set:
            if q >= l:
                r[q] += gs[int(l), q] * s[q - int(l)]

    r = channel_effects_np(r, sigma_2, channel_type)
    Y_tilda = np.reshape(r, [M, N], order='F')
    Y = Y_tilda @ Fn

    y_vec = Y.T.reshape((N * M,), order='F')
    data_index = np.nonzero(np.reshape(data_grid, (N * M,), order='F') > 0)[0]
    y_data = y_vec[data_index]

    if args.doppler_type == 'none':
        return y_data[:data.shape[0]]
    elif args.doppler_type == 'multi':
        nu_ml_tilda = gen_delay_time_channel_vectors(N, M, l_max, gs)
        H_tf = generate_time_frequency_channel_zp(N, M, gs, L_set)

        n_ite = 50
        decision = 1
        init_estimate = 1

        if args.compensation_type == 'lmmse':
            est_info_bits_LMMSE, eq_data = block_LMMSE_detector(
                N, M, M_mod, sigma_2, data_grid, r, gs, L_set, demodulate_fn)
        elif args.compensation_type == 'mrc':
            est_info_bits_MRC, det_iters_MRC, eq_data = mrc_delay_time_detector(
                N, M, M_data, M_mod, sigma_2, data_grid, r, H_tf, nu_ml_tilda,
                L_set, omega, decision, init_estimate, n_ite, modulate_fn, demodulate_fn)
        elif args.compensation_type == 'mrc_low_complexity':
            est_info_bits_MRC, det_iters_MRC, eq_data = mrc_low_complexity(
                N, M, M_mod, sigma_2, data_grid, r, H_tf, gs, L_set,
                omega, decision, init_estimate, n_ite, demodulate_fn, modulate_fn)
        elif args.compensation_type == 'single_tap':
            est_info_bits_1tap, eq_data = TF_single_tap_equalizer(
                N, M, M_mod, sigma_2, data_grid, Y, H_tf, demodulate_fn)
        elif args.compensation_type == 'none':
            eq_data = y_data
        else:
            raise ValueError("Invalid compensation type")
        eq_data = eq_data[:len(data)]
    return eq_data

def apply_channel_effects(symbols, snr_eval, channel_type, modulate_fn, demodulate_fn, args):
    batch_size = args.batch_size
    M = args.M_number
    N = args.N_number
    div_num = N * (M / 16)
    len_syms = int(M * N - div_num)

    symbols = tf.reshape(symbols, [batch_size, -1])
    symbol_length = tf.shape(symbols)[1]

    num_otfs_frames = int(tf.math.ceil(symbol_length / len_syms).numpy())
    total_symbols_needed = num_otfs_frames * len_syms

    pad_len = int(total_symbols_needed - symbol_length.numpy())
    padded_symbols = tf.pad(symbols, [[0, 0], [0, pad_len]], "CONSTANT", constant_values=0)

    otfs_frames = tf.reshape(padded_symbols, [batch_size, num_otfs_frames, len_syms])

    if args.doppler_type == 'multi':
        noisy_otfs_frames = np.array([
            np.array([apply_doppler_channel(frame, snr_eval, channel_type, modulate_fn, demodulate_fn, args)
                      for frame in batch_frames])
            for batch_frames in otfs_frames
        ])
    elif args.doppler_type == 'single':
        # Not used in the original code but kept for completeness.
        noisy_otfs_frames = np.array([
            np.array([apply_doppler_channel(frame, snr_eval, channel_type, modulate_fn, demodulate_fn, args)
                      for frame in batch_frames])
            for batch_frames in otfs_frames
        ])
    elif args.doppler_type == 'none':
        noisy_otfs_frames = np.array([
            np.array([handle_none_mode(frame, snr_eval, channel_type)
                      for frame in batch_frames])
            for batch_frames in otfs_frames
        ])

    noisy_symbols = np.stack(noisy_otfs_frames, axis=0)
    noisy_symbols = tf.reshape(noisy_symbols, [batch_size, num_otfs_frames * len_syms])
    noisy_symbols = noisy_symbols[:, :symbol_length]
    noisy_symbols = tf.reshape(noisy_symbols, [-1])
    return noisy_symbols

def get_channel_coding_options(args):
    mode = args.channel_coding_mode.lower()
    if mode == "true":
        return [True]
    elif mode == "false":
        return [False]
    else:
        return [True, False]

def calculate_ber(code_bits, received_bits):
    code_bits = tf.cast(code_bits, tf.float32)
    received_bits = tf.cast(received_bits, tf.float32)
    errors = tf.cast(tf.not_equal(code_bits, received_bits), tf.float32)
    ber_per_batch = tf.reduce_mean(errors, axis=1)
    avg_ber = tf.reduce_mean(ber_per_batch)
    return avg_ber

def test_ber(model, args, channel_type):
    snr_boundaries = [0, 5, 12, 20, 26, 30]
    modulation_schemes = [
        {'modulation_order': 2, 'modulate_fn': modulate_psk, 'demodulate_fn': demodulate_psk},
        {'modulation_order': 4, 'modulate_fn': modulate_psk, 'demodulate_fn': demodulate_psk},
        {'modulation_order': 16, 'modulate_fn': modulate_qam, 'demodulate_fn': demodulate_qam},
        {'modulation_order': 64, 'modulate_fn': modulate_qam, 'demodulate_fn': demodulate_qam},
        {'modulation_order': 256, 'modulate_fn': modulate_qam, 'demodulate_fn': demodulate_qam},
    ]

    EbNo_list = [i for i in range(args.snr_min, args.snr_max, 2)]
    is_channel_coding_options = get_channel_coding_options(args)
    input_bits_len = args.input_bits_len

    coding_labels = {True: 'With Channel Coding', False: 'Without Channel Coding'}
    ber_results = {label: {} for label in coding_labels.values()}

    for is_cc in is_channel_coding_options:
        for EbNo_value in EbNo_list:
            code_bits = tf.random.uniform([args.batch_size, input_bits_len], minval=0, maxval=2, dtype=tf.int32)

            modulation_index = determine_modulation_index(args, EbNo_value, snr_boundaries)
            modulation_scheme = modulation_schemes[modulation_index]
            modulation_order = modulation_scheme['modulation_order']
            modulate_fn = modulation_scheme['modulate_fn']
            demodulate_fn = modulation_scheme['demodulate_fn']
            args.modulation_index = modulation_index

            snr = calculate_snr(EbNo_value, modulation_order, is_cc)
            snr_eval = tf.fill([args.batch_size, 1], value=snr)
            noiseVar = 1 / (10 ** (snr / 10))

            M_mod = modulation_order

            if is_cc:
                encoder = TurboEncoder(constraint_length=4, rate=1/3, terminate=True)
                decoder = TurboDecoder(
                    encoder,
                    num_iter=5,
                    algorithm="maxlog",
                    hard_out=False
                )

                code_bits_np = np.reshape(code_bits.numpy(), [args.batch_size, -1])
                encoded_bits = encoder.encode(code_bits_np)
                encoded_bits = tf.reshape(encoded_bits, [args.batch_size, -1])

                padding_bits, symbols = handle_channel_coding(encoded_bits, M_mod, model, modulate_fn)
            else:
                symbols = perform_modulate(code_bits, M_mod, modulate_fn)
                symbols = tf.cast(symbols, dtype=tf.complex64)
                padding_bits = 0

            noisy_symbols = apply_channel_effects(symbols, snr_eval, channel_type, modulate_fn, demodulate_fn, args)

            if is_cc:
                received_bits = perform_soft_demodulate(noisy_symbols, M_mod, demodulate_fn, demod_type='soft', noise_var=noiseVar)
                received_bits = tf.cast(received_bits, dtype=tf.float32)
                received_bits = tf.reshape(received_bits, [-1, ])
                received_bits = np.reshape(received_bits, [args.batch_size, -1])

                if padding_bits > 0:
                    received_bits = received_bits[:, : -padding_bits]

                # Turbo decode
                decoded_concat = []
                for i in range(args.batch_size):
                    dec_result = decoder.decode(received_bits[i:i+1, :]).flatten()
                    dec_bits_int = (dec_result > 0.0).astype(np.int32)
                    decoded_concat.append(dec_bits_int)
                decoded_concat = np.array(decoded_concat)

                # slice to input_bits_len
                decoded_concat = decoded_concat[:, :input_bits_len]
                # shape is [batch_size, input_bits_len]

                # Convert back to TF for BER calculation
                final_decoded = tf.constant(decoded_concat, dtype=tf.int32)
                code_bits_tf = tf.constant(code_bits_np, dtype=tf.int32)
                # shape is [batch_size, input_bits_len]
                ber = calculate_ber(code_bits_tf, final_decoded)

            else:
                received_bits = perform_demodulate(noisy_symbols, M_mod, demodulate_fn)
                received_bits = tf.cast(received_bits, dtype=tf.int32)

                code_bits_2d = tf.reshape(code_bits, shape=[args.batch_size, -1])
                received_bits_2d = tf.reshape(received_bits, shape=[args.batch_size, -1])
                ber = calculate_ber(code_bits_2d, received_bits_2d)

            ber_label = coding_labels[is_cc]
            ber_results[ber_label][EbNo_value] = ber.numpy()
            print(f'[BER] Channel={channel_type}, {ber_label}, EbNo={EbNo_value} dB, BER={ber_results[ber_label][EbNo_value]:.6f}')

    plot_ber_vs_snr(ber_results, channel_type)


def plot_ber_vs_snr(ber_results, channel_type):
    os.makedirs('plot_results', exist_ok=True)
    plt.figure(figsize=(10, 6))

    for label, results in ber_results.items():
        snr_values = sorted(results.keys())
        ber_values = [results[snr] for snr in snr_values]
        plt.semilogy(snr_values, ber_values, marker='o', label=label)

    plt.title(f"BER vs EbNo for {channel_type.upper()} Channel")
    plt.xlabel("EbNo (dB)")
    plt.ylabel("BER (Bit Error Rate)")
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plot_results/{channel_type}_ber_result_with_and_without_channel_coding.png", dpi=400)


def test_psnr(test_dataset, model, args, channel_type):
    snr_boundaries = [0, 5, 12, 20, 26, 30]

    modulation_schemes = [
        {'modulation_order': 2, 'modulate_fn': modulate_psk, 'demodulate_fn': demodulate_psk},
        {'modulation_order': 4, 'modulate_fn': modulate_psk, 'demodulate_fn': demodulate_psk},
        {'modulation_order': 16, 'modulate_fn': modulate_qam, 'demodulate_fn': demodulate_qam},
        {'modulation_order': 64, 'modulate_fn': modulate_qam, 'demodulate_fn': demodulate_qam},
        {'modulation_order': 256, 'modulate_fn': modulate_qam, 'demodulate_fn': demodulate_qam},
    ]

    sample_images = next(iter(test_dataset))
    EbNo_list = [i for i in range(args.snr_min, args.snr_max, 2)]

    is_channel_coding_options = [True, False]
    input_bits_len = args.input_bits_len
    coding_labels = {True: 'With Channel Coding', False: 'Without Channel Coding'}
    psnr_results = {label: {} for label in coding_labels.values()}

    for is_cc in is_channel_coding_options:
        for EbNo_value in EbNo_list:
            modulation_index = determine_modulation_index(args, EbNo_value, snr_boundaries)
            modulation_scheme = modulation_schemes[modulation_index]
            modulation_order = modulation_scheme['modulation_order']
            modulate_fn = modulation_scheme['modulate_fn']
            demodulate_fn = modulation_scheme['demodulate_fn']
            args.modulation_index = modulation_index

            snr_value = calculate_snr(EbNo_value, modulation_order, is_cc)
            snr_eval = tf.fill([args.batch_size, 1], value=snr_value)

            noiseVar = 1 / (10 ** (snr_value / 10))

            # Encode
            quantized, z_shape, code_indices, flat_inputs = model.encode(
                sample_images,
                tf.constant(modulation_index, dtype=tf.int32),
                training=False)

            code_bits = model.code2bit(code_indices)  # shape (B*H*W, log2(num_embeddings))
            code_bits = tf.cast(code_bits, tf.float32)

            M_mod = modulation_order
            if is_cc:
                code_bits_int = tf.cast(code_bits, tf.int32)

                encoder = TurboEncoder(constraint_length=4, rate=1/3, terminate=True)
                decoder = TurboDecoder(encoder, num_iter=5, algorithm="maxlog", hard_out=False)

                code_bits_np = np.reshape(code_bits_int.numpy(), [args.batch_size, -1])
                encoded_bits = encoder.encode(code_bits_np)
                encoded_bits = tf.reshape(encoded_bits, [args.batch_size, -1])

                padding_bits, symbols = handle_channel_coding(encoded_bits, modulation_order, model, modulate_fn)
            else:
                symbols = perform_modulate(code_bits, M_mod, modulate_fn)
                symbols = tf.cast(symbols, dtype=tf.complex64)
                padding_bits = 0

            # Channel
            noisy_symbols = apply_channel_effects(symbols, snr_eval, channel_type, modulate_fn, demodulate_fn, args)

            # Demod
            if is_cc:
                received_bits = perform_soft_demodulate(
                    noisy_symbols, M_mod, demodulate_fn,
                    demod_type='soft', noise_var=noiseVar)
                received_bits = tf.cast(received_bits, dtype=tf.float32)
                received_bits = tf.reshape(received_bits, [-1, ])

                received_bits = np.reshape(received_bits, [args.batch_size, -1])

                if padding_bits > 0:
                    received_bits = received_bits[:, : -padding_bits]

                decoded_concat = []
                for i in range(args.batch_size):
                    dec_result = decoder.decode(received_bits[i:i+1, :]).flatten()
                    dec_bits_int = (dec_result > 0.0).astype(np.int32)
                    decoded_concat.append(dec_bits_int)
                decoded_concat = np.array(decoded_concat)

                decoded_concat = decoded_concat[:, :input_bits_len]
                recovered_indices = model.bit2code(decoded_concat.flatten())
            else:
                received_bits = perform_demodulate(noisy_symbols, M_mod, demodulate_fn)
                received_bits = tf.cast(received_bits, dtype=tf.int32)
                recovered_indices = model.bit2code(received_bits)

            mapping_vector = model.vq_layer.embed_code(
                recovered_indices,
                [z_shape[0], z_shape[1], z_shape[2], z_shape[3]])

            reconstructions = model.decode(mapping_vector, modulation_index, z_shape, training=False)

            psnr_values = tf.image.psnr(sample_images, reconstructions, max_val=1.0)
            mean_psnr = tf.reduce_mean(psnr_values)
            psnr_label = coding_labels[is_cc]
            psnr_results[psnr_label][EbNo_value] = mean_psnr.numpy()
            print(f'[PSNR] Channel={channel_type}, {psnr_label}, EbNo={EbNo_value} dB, PSNR={psnr_results[psnr_label][EbNo_value]:.6f}')

    plot_psnr_vs_snr(psnr_results, channel_type)

def plot_psnr_vs_snr(psnr_results, channel_type):
    os.makedirs('plot_results', exist_ok=True)

    plt.figure(figsize=(10, 6))

    for label, results in psnr_results.items():
        snr_values = sorted(results.keys())
        psnr_values = [results[snr] for snr in snr_values]
        plt.semilogy(snr_values, psnr_values, marker='o', label=label)

    plt.title(f"PSNR vs EbNo for {channel_type.upper()} Channel")
    plt.xlabel("EbNo (dB)")
    plt.ylabel("PSNR")
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plot_results/{channel_type}_psnr_result_with_and_without_channel_coding.png", dpi=400)

def main_with_args(args):
    num_modulations = len(MODULATION_ORDERS)
    model = VQVAE(
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
        num_modulations=num_modulations,
        commitment_cost=args.commitment_cost,
        decay=args.decay,
        n_res_block=args.n_res_block
    )

    if args.eval_type.lower() == "psnr":
        from keras.datasets import cifar10
        (x_train, _), (x_test, _) = cifar10.load_data()
        x_train = preprocess_data(x_train)
        x_test = preprocess_data(x_test)
        _, x_val = train_test_split(x_train, test_size=0.1, shuffle=True, random_state=42)

        test_dataset = tf.data.Dataset.from_tensor_slices((x_test))\
            .batch(args.batch_size, drop_remainder=True)\
            .prefetch(tf.data.experimental.AUTOTUNE)
        model = load_models_from_dir(model)
        test_psnr(test_dataset, model, args, args.channel_type)
    else:
        test_ber(model, args, args.channel_type)

def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_type", type=str, choices=["ber", "psnr"], default="ber",
                        help="Select 'ber' or 'psnr' evaluation.")
    parser.add_argument("--channel_coding_mode", type=str, choices=["both", "true", "false"], default="both",
                        help="Choose if channel coding is used: 'true', 'false', or 'both'.")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=400)
    parser.add_argument("--lr_max", type=float, default=1e-3)
    parser.add_argument("--lr_min", type=float, default=5e-5)
    parser.add_argument("--cycle_length", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_embeddings", type=int, default=512)
    parser.add_argument("--commitment_cost", type=float, default=0.25)
    parser.add_argument("--decay", type=float, default=0.99)
    parser.add_argument("--n_res_block", type=int, default=2)
    parser.add_argument("--n_res_channel", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--input_bits_len", type=int, default=2304)
    parser.add_argument("--random_seed", type=int, default=128)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--snr_db", type=float, default=10.0)
    parser.add_argument("--modulation", type=str,
                        choices=['auto', 'BPSK', 'QPSK', '16QAM', '64QAM', '256QAM'],
                        default="auto")
    parser.add_argument("--M_number", type=int, default=64)
    parser.add_argument("--N_number", type=int, default=16)
    parser.add_argument("--snr_min", type=int, default=-10)
    parser.add_argument("--snr_max", type=int, default=30)
    parser.add_argument("--channel_type", type=str, choices=['awgn', 'rayleigh', 'rician'], default='awgn')
    parser.add_argument("--doppler_type", type=str, choices=['multi', 'none'], default='none',
                        help="도플러 적용 방식")
    parser.add_argument("--compensation_type", type=str,
                        choices=['lmmse', 'single_tap', 'mrc', 'mrc_low_complexity', 'none'],
                        default='lmmse', help="도플러 보상 방식")
    args = parser.parse_args()
    main_with_args(args)


if __name__ == '__main__':
    main_cli()
