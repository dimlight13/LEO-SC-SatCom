import argparse
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import load_models_from_dir, load_config
import tensorflow_datasets as tfds
from utils import val_preprocessing
from models import VQVAE, Actor, PostLMMSENet
from modulate_fn_np import demodulate_psk, modulate_psk
from modulate_fn_np import modulate_qam, demodulate_qam
from modulate_fn_np import perform_modulate, perform_demodulate, perform_soft_demodulate
from utils import channel_effects_np
from Doppler_utils import dft_matrix, generate_2d_data_grid, generate_delay_Doppler_channel_parameters
from Doppler_utils import mrc_delay_time_detector, block_LMMSE_detector, apply_channel
from Doppler_utils import gen_discrete_time_channel, gen_delay_time_channel_vectors, generate_time_frequency_channel_zp
import tensorflow_probability as tfp
from ldpc_fn_tf import LDPC5GEncoder, LDPC5GDecoder

MODULATION_ORDERS = [2, 4, 16, 64, 256]

BITS_PER_SYMBOL = {
    2: 1,   # BPSK
    4: 2,   # QPSK
    16: 4,  # 16-QAM
    64: 6,  # 64-QAM
    256: 8  # 256-QAM
}

def determine_modulation_index(args, EbNo, snr_boundaries):
    if args.modulation == 'TN_auto':
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

def apply_doppler_channel(frame, snr_eval, channel_type, modulate_fn, demodulate_fn, args):
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

    max_speed = 480  # km/hr
    chan_coef, delay_taps, Doppler_taps, taps = generate_delay_Doppler_channel_parameters(
        N, M, car_fre, delta_f, T, max_speed, args.profile)
    L_set = np.unique(delay_taps)

    gs = gen_discrete_time_channel(N, M, taps, delay_taps, Doppler_taps, chan_coef)

    r = np.zeros(N * M, dtype=complex)
    l_max = int(max(delay_taps))

    r = apply_channel(N, M, gs, s, L_set)

    r = channel_effects_np(r, sigma_2, channel_type)
    Y_tilda = np.reshape(r, [M, N], order='F')
    Y = Y_tilda @ Fn

    y_vec = Y.T.reshape((N * M,), order='F')
    data_index = np.nonzero(np.reshape(data_grid, (N * M,), order='F') > 0)[0]
    y_data = y_vec[data_index]

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

    noisy_otfs_frames = np.array([
        np.array([apply_doppler_channel(frame, snr_eval, channel_type, modulate_fn, demodulate_fn, args)
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

def resize(inputs, target_size):
    return tf.image.resize(inputs, target_size)

def test_psnr(test_dataset, model, tx_agent, post_model, args, channel_type):
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

    chan_coef, delay_taps, Doppler_taps, _ = generate_delay_Doppler_channel_parameters(
        16, 64, 20e9, 15e3, 1 / 15e3, 480, args.profile)

    powers = np.abs(chan_coef)**2
    norm_p = powers / np.sum(powers)
    mu_tau = np.sum(norm_p * delay_taps)
    mu_fd = np.sum(norm_p * Doppler_taps)

    tau_rms = np.float32(np.sqrt(np.sum(norm_p * (delay_taps - mu_tau)**2)))
    fd_rms = np.float32(np.sqrt(np.sum(norm_p * (Doppler_taps - mu_fd)**2)))

    tau_rms_all = tf.fill([args.batch_size, 5], value=tau_rms)
    fd_rms_all = tf.fill([args.batch_size, 5], value=fd_rms)
    
    if args.channel_coding_mode == 'both':
        is_channel_coding_options = [True, False]
    elif args.channel_coding_mode == 'true':
        is_channel_coding_options = [True]
    elif args.channel_coding_mode == 'false':
        is_channel_coding_options = [False]

    # is_channel_coding_options = [True, False]
    input_bits_len = args.input_bits_len
    coding_labels = {True: 'With Channel Coding', False: 'Without Channel Coding'}
    psnr_results = {label: {} for label in coding_labels.values()}
    
    inputs_agent = tf.image.resize(sample_images, (8,8))

    for is_cc in is_channel_coding_options:
        for EbNo_value in EbNo_list:
            EbNo_eval = tf.fill([args.batch_size, 1], value=EbNo_value)
            EbNo_eval = tf.cast(EbNo_eval, dtype=tf.float32)

            if args.modulation == 'SC_auto' or args.modulation == 'SC_none':
                logits = tx_agent([inputs_agent, EbNo_eval, tau_rms_all, fd_rms_all], training=False)              # [B,5]
                dist   = tfp.distributions.Categorical(logits=logits)
                modulation_index = dist.sample()                                         # [B]
                modulation_index = modulation_index[0]
            else:
                modulation_index = determine_modulation_index(args, EbNo_value, snr_boundaries)

            modulation_scheme = modulation_schemes[modulation_index]
            modulation_order = modulation_scheme['modulation_order']
            modulate_fn = modulation_scheme['modulate_fn']
            demodulate_fn = modulation_scheme['demodulate_fn']
            args.modulation_index = modulation_index

            snr_value = calculate_snr(EbNo_value, modulation_order, is_cc)
            snr_eval = tf.fill([args.batch_size, 1], value=snr_value)

            noiseVar = 1 / (10 ** (snr_value / 10))

            encoded_features, z_shape = model.encode(sample_images, tf.constant(modulation_index, dtype=tf.int32), training=False)
            quantized, code_indices, flat_inputs = model.quantize(encoded_features, training=False)

            code_bits = model.code2bit(code_indices)  # shape (B*H*W, log2(num_embeddings))
            code_bits = tf.cast(code_bits, tf.float32)
            M_mod = modulation_order

            group_size = BITS_PER_SYMBOL[M_mod] 
            original_length = tf.shape(code_bits)[-1]
            remainder = tf.math.floormod(original_length, group_size)

            pad_size = tf.cond(
                tf.equal(remainder, 0),
                lambda: 0,
                lambda: group_size - remainder
            )

            if is_cc:
                code_bits_int = tf.cast(code_bits, tf.int32)

                # encoder = TurboEncoder(constraint_length=4, rate=1/3, terminate=True)
                # decoder = TurboDecoder(encoder, num_iter=5, algorithm="maxlog", hard_out=False)
                n_num = int(args.input_bits_len * 3)
                encoder = LDPC5GEncoder(k=args.input_bits_len, n=n_num)
                decoder = LDPC5GDecoder(encoder, num_iter=5)

                code_bits_np = np.reshape(code_bits_int.numpy(), [args.batch_size, -1])
                encoded_bits = encoder.encode(code_bits_np)
                encoded_bits = tf.reshape(encoded_bits, [args.batch_size, -1])

                padding_bits, symbols = handle_channel_coding(encoded_bits, modulation_order, model, modulate_fn)
            else:
                code_bits_padded = tf.pad(code_bits, [[0, 0], [0, pad_size]], constant_values=0)
                padded_length = tf.shape(code_bits_padded)[-1]

                symbols = perform_modulate(code_bits_padded, M_mod, modulate_fn)
                symbols = tf.cast(symbols, dtype=tf.complex64)
                padding_bits = 0
            
            sym_len = tf.reshape(symbols, [args.batch_size, -1]).shape[1]

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
                if args.modulation == 'SC_auto':
                    noisy_symbols = tf.reshape(noisy_symbols, [args.batch_size, -1])
                    
                    prev_len = sym_len
                    eq_batch = tf.pad(noisy_symbols, [[0, 0], [0, args.input_bits_len - prev_len]], constant_values=0)

                    modulation_index_batch  = tf.fill([args.batch_size, 1], value=modulation_index)

                    snr_value_per = EbNo_eval[0][0]

                    EbNo_eval = tf.reshape(EbNo_eval, (-1, ))
                    eq_symbols = post_model(eq_batch, modulation_index_batch, training=False)
                    eq_symbols = eq_symbols[:, :sym_len] 
                    noisy_symbols = tf.reshape(eq_symbols, (-1,))

                received_bits = perform_demodulate(noisy_symbols, M_mod, demodulate_fn)
                received_bits = tf.cast(received_bits, dtype=tf.int32)
                received_bits = tf.reshape(received_bits, (-1, padded_length))
                received_bits = received_bits[:, :9]
                recovered_indices = model.bit2code(received_bits)

            mapping_vector = model.vq_layer.embed_code(
                recovered_indices,
                [z_shape[0], z_shape[1], z_shape[2], args.embedding_dim])

            reconstructions = model.decode(mapping_vector, modulation_index, z_shape, training=False)

            psnr_values = tf.image.psnr(sample_images, reconstructions, max_val=1.0)
            mean_psnr = tf.reduce_mean(psnr_values)
            psnr_label = coding_labels[is_cc]
            psnr_results[psnr_label][EbNo_value] = mean_psnr.numpy()
            print(f'[PSNR] Channel={channel_type}, {psnr_label}, EbNo={EbNo_value} dB, PSNR={psnr_results[psnr_label][EbNo_value]:.6f}')

    plot_psnr_vs_snr(psnr_results, args)

def plot_psnr_vs_snr(psnr_results, args):
    os.makedirs(f'plot_results/{args.dataset_name}', exist_ok=True)

    plt.figure(figsize=(10, 6))

    for label, results in psnr_results.items():
        snr_values = sorted(results.keys())
        psnr_values = [results[snr] for snr in snr_values]
        plt.semilogy(snr_values, psnr_values, marker='o', label=label)

    plt.title(f"PSNR vs EbNo for {args.dataset_name} ({args.modulation})")
    plt.ylim(20, 35)
    plt.xlabel("EbNo (dB)")
    plt.ylabel("PSNR")
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plot_results/{args.dataset_name}/psnr_vs_snr_{args.modulation}.png", dpi=400)

def load_grouped_tfrecord(filename: str, batch_size: int = 32):
    feature_description = {
        'psnr_all':    tf.io.FixedLenFeature([], tf.string),
        'tau_rms_all': tf.io.FixedLenFeature([], tf.string),
        'fd_rms_all':  tf.io.FixedLenFeature([], tf.string),
        'snr':         tf.io.FixedLenFeature([], tf.int64),
    }
    def _parse_fn(proto):
        parsed     = tf.io.parse_single_example(proto, feature_description)
        psnr_all   = tf.io.parse_tensor(parsed['psnr_all'],    out_type=tf.float32)
        tau_rms_all= tf.io.parse_tensor(parsed['tau_rms_all'], out_type=tf.float32)
        fd_rms_all = tf.io.parse_tensor(parsed['fd_rms_all'],  out_type=tf.float32)
        snr        = tf.cast(parsed['snr'], tf.float32)
        return {
            'psnr_all':    psnr_all,
            'tau_rms_all': tau_rms_all,
            'fd_rms_all':  fd_rms_all,
            'snr':         tf.expand_dims(snr, -1)
        }

    ds = tf.data.TFRecordDataset(filename)
    ds = ds.map(_parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def main_with_args(args):
    if not hasattr(args, "save_model_dir"):
        config = load_config(f"config/{args.dataset_name}/evaluation_config.yaml")
        args.save_model_dir = config.get('pretrain_vqvae_model_dir', f'./vqvae_model/{args.dataset_name}')
        args.num_modulations = config.get('num_modulations', 5)
        args.num_embeddings = config.get('num_embeddings', 512)
        args.commitment_cost = config.get('commitment_cost', 0.25)
        args.decay = config.get('decay', 0.99)
        args.n_res_block = config.get('n_res_block', 2)
        args.embedding_dim = config.get('embedding_dim', 32)
        args.dataset_name = config.get('dataset_name', f'{args.dataset_name}')

    num_modulations = len(MODULATION_ORDERS)
    img_size = args.img_size

    model = VQVAE(
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
        num_modulations=num_modulations,
        commitment_cost=args.commitment_cost,
        decay=args.decay,
        n_res_block=args.n_res_block,
        img_size=img_size,
    )

    if args.dataset_name == 'cifar10':
        x_test = tfds.load(args.dataset_name, split="test", with_info=False, shuffle_files=False)
    elif args.dataset_name == 'eurosat':
        x_test = tfds.load(args.dataset_name, split="train[90%:]", with_info=False, shuffle_files=True)

    test_dataset = (x_test
            .map(lambda x: val_preprocessing(x, img_size), num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(args.batch_size * 2, seed=42, reshuffle_each_iteration=False)
            .batch(args.batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
            )

    save_model_dir = args.save_model_dir
    save_rl_model_dir = args.save_rl_model_dir
    post_model_dir = args.post_model_dir

    tx_agent = Actor(5)
    input_agent_dummy = tf.zeros((1, 8, 8, 3), dtype=tf.float32)
    tau_rms_dummy = tf.zeros((1, 5), dtype=tf.float32)
    fd_rms_dummy = tf.zeros((1, 5), dtype=tf.float32)
    EbNo_eval_dummy = tf.zeros((1, 1), dtype=tf.float32)
    _ = tx_agent([input_agent_dummy, EbNo_eval_dummy, tau_rms_dummy, fd_rms_dummy])
    tx_agent.load_weights(os.path.join(save_rl_model_dir, 'tx_actor.h5'))

    post_model = PostLMMSENet()
    eq_batch_dummy = tf.zeros((1, 2304), dtype=tf.float32)
    mod_idx_batch_dummy = tf.zeros((1, 1), dtype=tf.int32)
    snr_val_dummy = tf.zeros((1, ), dtype=tf.float32)

    _ = post_model(eq_batch_dummy, mod_idx_batch_dummy)
    post_model.load_weights(os.path.join(post_model_dir, 'post_model.h5'))

    model = load_models_from_dir(save_model_dir, model)
    test_psnr(test_dataset, model, tx_agent, post_model, args, args.channel_type)

def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='eurosat', help="Name of the dataset to use", choices=['cifar10', 'eurosat'])
    args = parser.parse_args()
    
    parser.add_argument("--config", type=str, default=f"config/{args.dataset_name}/model_config.yaml", help="Path to the config file")
    args = parser.parse_args()
    config = load_config(args.config)

    parser.add_argument("--save_model_dir", type=str, default=config.get('pretrain_vqvae_model_dir', f'./vqvae_model/{args.dataset_name}'), help="Directory to save trained model")
    parser.add_argument("--save_rl_model_dir", type=str, default=config.get('rl_rx_agent_model_dir', f'./rl_model/{args.dataset_name}'), help="Directory to save trained RL model")
    parser.add_argument("--post_model_dir", type=str, default=config.get('post_model_dir', f'./post_model/{args.dataset_name}'))
    parser.add_argument("--num_modulations", type=int, default=config.get('num_modulations', 5))
    parser.add_argument("--num_embeddings", type=int, default=config.get('num_embeddings', 512))
    parser.add_argument("--commitment_cost", type=float, default=config.get('commitment_cost', 0.25))
    parser.add_argument("--decay", type=float, default=config.get('decay', 0.99))
    parser.add_argument("--n_res_block", type=int, default=config.get('n_res_block', 2))
    parser.add_argument("--embedding_dim", type=int, default=config.get('embedding_dim', 32))
    parser.add_argument("--channel_coding_mode", type=str, choices=["both", "true", "false"], default="both",
                        help="Choose if channel coding is used: 'true', 'false', or 'both'.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--input_bits_len", type=int, default=2304)
    parser.add_argument("--modulation", type=str,
                        choices=['TN_auto', 'SC_auto', 'SC_none', 'BPSK', 'QPSK', '16QAM', '64QAM', '256QAM'],
                        default="SC_auto")
    parser.add_argument("--img_size", type=int,  default=config.get('img_size', 32))
    
    parser.add_argument("--M_number", type=int, default=64)
    parser.add_argument("--N_number", type=int, default=16)
    parser.add_argument("--snr_min", type=int, default=0)
    parser.add_argument("--snr_max", type=int, default=60)
    parser.add_argument("--channel_type", type=str, choices=['awgn', 'rayleigh', 'rician'], default='awgn')
    parser.add_argument("--doppler_type", type=str, choices=['multi', 'none'], default='multi',
                        help="How Doppler is applied")
    parser.add_argument("--compensation_type", type=str,
                        choices=['lmmse', 'mrc', 'none'],
                        default='lmmse', help="Doppler compensation method")
    parser.add_argument("--profile", type=str, choices=['NTN-TDL-A', 'NTN-TDL-B', 'NTN-TDL-C', 'NTN-TDL-D'], default='NTN-TDL-D')
    args = parser.parse_args()
    main_with_args(args)


if __name__ == '__main__':
    main_cli()
