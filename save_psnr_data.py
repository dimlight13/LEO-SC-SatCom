from utils import channel_effects_np, train_preprocessing, val_preprocessing, load_models_from_dir
from models import VQVAE
from Doppler_utils import generate_delay_Doppler_channel_parameters, generate_2d_data_grid, gen_discrete_time_channel, dft_matrix
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import argparse
from utils import load_config
from tqdm import tqdm
from modulate_fn_tf import modulate_psk, demodulate_psk, modulate_qam, demodulate_qam
import numba
import os
import sys

tf.random.set_seed(42)

def file_exists_nonempty(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))): 
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def serialize_psnr(profile, psnr, tau_rms, fd_rms, snr, mod_idx):
    if isinstance(profile, str):
        profile_bytes = profile.encode('utf-8')
    else:
        profile_bytes = profile   # 이미 bytes

    psnr = tf.io.serialize_tensor(psnr)
    feature = {
        'profile': _bytes_feature(profile_bytes),
        'psnr': _bytes_feature(psnr),
        'tau_rms': _bytes_feature(tf.io.serialize_tensor(tau_rms)),
        'fd_rms': _bytes_feature(tf.io.serialize_tensor(fd_rms)),
        'snr': _int64_feature(snr),
        'modulation_index': _int64_feature(int(mod_idx))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

MODULATION_ORDERS = [2, 4, 16, 64, 256]

BITS_PER_SYMBOL = {
    2: 1,   # BPSK
    4: 2,   # QPSK
    16: 4,  # 16-QAM
    64: 6,  # 64-QAM
    256: 8  # 256-QAM
}

modulation_schemes = [
    {'modulation_order': 2, 'modulate_fn': modulate_psk, 'demodulate_fn': demodulate_psk},
    {'modulation_order': 4, 'modulate_fn': modulate_psk, 'demodulate_fn': demodulate_psk},
    {'modulation_order': 16, 'modulate_fn': modulate_qam, 'demodulate_fn': demodulate_qam},
    {'modulation_order': 64, 'modulate_fn': modulate_qam, 'demodulate_fn': demodulate_qam},
    {'modulation_order': 256, 'modulate_fn': modulate_qam, 'demodulate_fn': demodulate_qam},
]

@numba.njit
def block_LMMSE_detector(N, M, noise_var, data_grid, r, gs, L_set):
    Fn = dft_matrix(N)
    norm_Fn = np.linalg.norm(Fn, 2)
    Fn = (Fn / np.float32(norm_Fn)).astype(np.complex128)

    data_array = data_grid.T.ravel()
    data_index = np.where(data_array > 0)[0]

    sn_block_est = np.zeros((M, N), dtype=np.complex128)

    for n in range(N):
        Gn = np.zeros((M, M), dtype=np.complex128)
        for m in range(M):
            for l in (L_set + 1): 
                if (m + 1) >= l:
                    Gn[m, (m + 1) - int(l)] = gs[int(l) - 1, m + n * M]
        rn = r[n * M: (n + 1) * M]
        Rn = np.dot(np.conj(Gn).T, Gn)
        sn_block_est[:, n] = np.linalg.inv(Rn + noise_var * np.eye(M)) @ (np.dot(np.conj(Gn).T, rn))

    X_tilda_est = sn_block_est
    X_est = np.dot(X_tilda_est, Fn)
    x_est = X_est.T.ravel()
    x_data = x_est[data_index]
    return x_data

def apply_doppler_channel(frame, snr_eval, modulation_index, profile):
    if isinstance(profile, (bytes, np.bytes_)):
        profile = profile.decode('utf-8')

    if modulation_index == 0:
        M_mod = 2
    elif modulation_index == 1:
        M_mod = 4
    elif modulation_index == 2:
        M_mod = 16
    elif modulation_index == 3:
        M_mod = 64
    elif modulation_index == 4:
        M_mod = 256
    else:
        raise ValueError("Invalid modulation type")

    M = 64
    N = 16

    length_ZP = M / 16
    M_data = int(M - length_ZP)
    data_grid = np.zeros((M, N), dtype=np.float32)
    data_grid[:M_data, :] = 1

    snr_dB = snr_eval

    car_fre = 20e9
    delta_f = 15e3
    T = 1 / delta_f

    Fn = dft_matrix(N)
    norm_Fn = np.linalg.norm(Fn, 2)
    Fn = (Fn / np.float32(norm_Fn)).astype(np.complex128)

    eng_sqrt = 1 if M_mod == 2 else np.sqrt((M_mod - 1) / 6 * 4)

    SNR_linear = 10 ** (snr_dB / 10)
    sigma_2 = (np.abs(eng_sqrt) ** 2) / SNR_linear
    sigma_2 = np.float32(sigma_2)

    data = frame

    X = generate_2d_data_grid(N, M, data, data_grid)

    X_tilda = X @ np.conjugate(Fn).T
    s = np.reshape(X_tilda, [N * M], order='F')

    max_speed = 480 

    chan_coef, delay_taps, Doppler_taps, taps = generate_delay_Doppler_channel_parameters(
        N, M, car_fre, delta_f, T, max_speed, profile)
    L_set = np.unique(delay_taps)

    gs = gen_discrete_time_channel(N, M, taps, delay_taps, Doppler_taps, chan_coef)

    powers = np.abs(chan_coef)**2
    norm_p = powers / np.sum(powers)
    mu_tau = np.sum(norm_p * delay_taps)
    tau_rms = np.float32(np.sqrt(np.sum(norm_p * (delay_taps - mu_tau)**2)))
    mu_fd = np.sum(norm_p * Doppler_taps)
    fd_rms = np.float32(np.sqrt(np.sum(norm_p * (Doppler_taps - mu_fd)**2)))

    r = np.zeros(N * M, dtype=np.complex128)

    for q in range(N * M):
        for l in L_set:
            if q >= l:
                r[q] += gs[int(l), q] * s[q - int(l)]

    r = channel_effects_np(r, sigma_2)
    eq_data = block_LMMSE_detector(N, M, sigma_2, data_grid, r, gs, L_set)
    return eq_data[:data.shape[0]], tau_rms, fd_rms

@tf.function
def apply_doppler_channel_tf(frame, snr_eval, modulation_index, profile):
    noisy_frames, tau_rms, fd_rms = tf.numpy_function(
        func=apply_doppler_channel,
        inp=[frame, snr_eval, modulation_index, profile],
        Tout=(tf.complex128, tf.float32, tf.float32)
    )
    noisy_frames = tf.cast(noisy_frames, tf.complex64)
    return noisy_frames, tau_rms, fd_rms

def save_train_psnr(dataset, vqvae_model, args):
    len_syms = 960 # (64 (M) * 16 (N)) - (16 (zero_padded) * 4 (column_num)) = 1024 - 64 = 960
    batch_size = args.batch_size
    profile_list = ['NTN-TDL-A', 'NTN-TDL-B', 'NTN-TDL-C', 'NTN-TDL-D']

    num_batches = tf.data.experimental.cardinality(dataset).numpy()
    print(f"Total batches in dataset: {num_batches}")

    filename = os.path.join(args.save_dir, "doppler_psnr_train_dataset.tfrecord")
    if file_exists_nonempty(filename):
        print(f"[SKIP] Found existing train tfrecord: {filename}")
        return

    os.makedirs(args.save_dir, exist_ok=True)
    with tf.io.TFRecordWriter(filename) as writer:
        for profile in profile_list:
            print(f"Processing data for profile: {profile}")
            for mod in range(5):
                seed = tf.constant([123, 42], dtype=tf.int32)
                snr_train = tf.random.stateless_uniform(shape=[num_batches, args.batch_size, 1],
                                                        minval=0, maxval=60,
                                                        seed=seed, dtype=tf.int32)
                ds_iter = iter(dataset.take(num_batches))
                
                step = 0
                for images in tqdm(ds_iter, total=num_batches, desc=f"Modulation {mod}"):
                    snr_train_val = snr_train[step]
                    step += 1
                    modulation_index = tf.constant(mod, dtype=tf.int32)
                    M_mod = modulation_schemes[int(modulation_index)]['modulation_order']
                    modulate_fn = modulation_schemes[int(modulation_index)]['modulate_fn']
                    demodulate_fn = modulation_schemes[int(modulation_index)]['demodulate_fn']

                    encoded_features, z_shape = vqvae_model.encode(images, modulation_index, training=False)
                    _, code_indices, _ = vqvae_model.quantize(encoded_features, training=False)
                    code_bits = vqvae_model.code2bit(code_indices)
                    code_bits = tf.cast(code_bits, tf.float32)

                    group_size = BITS_PER_SYMBOL[M_mod]
                    original_length = tf.shape(code_bits)[-1]
                    remainder = tf.math.floormod(original_length, group_size)
                    pad_size = tf.cond(
                        tf.equal(remainder, 0),
                        lambda: 0,
                        lambda: group_size - remainder
                    )
                    code_bits_padded = tf.pad(code_bits, [[0, 0], [0, pad_size]], constant_values=0)

                    symbols = vqvae_model.modulate(code_bits_padded, M_mod, modulate_fn)
                    symbols_re = tf.reshape(symbols, [batch_size, -1])
                    sym_len = symbols_re.shape[1]

                    bits_per_symbol = tf.math.log(tf.cast(M_mod, tf.float32)) / tf.math.log(2.0)
                    num_symbols = 2304 / bits_per_symbol 
                    num_frames = tf.cast(tf.math.ceil(num_symbols / len_syms), tf.int32)
                    total_symbols_needed = num_frames * len_syms
                    pad_len = total_symbols_needed - sym_len

                    padded_length = tf.case([
                        (tf.equal(modulation_index, 0), lambda: tf.constant(9)),
                        (tf.equal(modulation_index, 1), lambda: tf.constant(10)),
                        (tf.equal(modulation_index, 2), lambda: tf.constant(12)),
                        (tf.equal(modulation_index, 3), lambda: tf.constant(12)),
                    ], default=lambda: tf.constant(16), exclusive=True)

                    padded_symbols = tf.pad(symbols_re, [[0, 0], [0, pad_len]], "CONSTANT", constant_values=0)
                    frames = tf.reshape(padded_symbols, [batch_size, num_frames, len_syms])

                    snr_per_sample = tf.reshape(snr_train_val, [batch_size])

                    noisy_frames, tau_rms, fd_rms = tf.map_fn(
                        lambda args: tf.map_fn(
                            lambda frame: apply_doppler_channel_tf(
                                frame,
                                args[1],    
                                modulation_index,
                                profile
                            ),
                            args[0],            
                            fn_output_signature=(
                                tf.complex64, tf.float32, tf.float32
                            )
                        ),
                        (frames, snr_per_sample), 
                        fn_output_signature=(
                            tf.complex64, tf.float32, tf.float32
                        )
                    )
                    noisy_symbols = tf.reshape(noisy_frames, [batch_size, -1])[:, :sym_len]

                    noisy_symbols = tf.reshape(noisy_symbols, (-1, ))
                    received_bits = vqvae_model.demodulate(noisy_symbols, M_mod, demodulate_fn)
                    received_bits = tf.cast(received_bits, tf.int32)
                    received_bits = tf.reshape(received_bits, (-1, padded_length))
                    received_bits = received_bits[:, :9]

                    recovered_indices = vqvae_model.bit2code(received_bits)
                    mapping_vector = vqvae_model.vq_layer.embed_code(
                        recovered_indices,
                        [batch_size, 16, 16, args.embedding_dim]
                    )
                    mapping_vector = tf.reshape(mapping_vector, (-1, 256, vqvae_model.embedding_dim))
                    recon = vqvae_model.decode(mapping_vector, modulation_index, [batch_size, 16, 16, 64], training=False)

                    psnr_val = tf.image.psnr(
                        tf.cast(images, tf.float32),
                        tf.cast(recon, tf.float32),
                        max_val=1.0
                    )
                    psnr_vals = psnr_val.numpy().tolist()    # [p0, p1, …, p127]
                    snr_vals  = snr_train_val.numpy().flatten().tolist()  # [s0, s1, …, s127]
                    tau_rmss = tau_rms.numpy().flatten().tolist()  # [t0, t1, …, t127]
                    fd_rmss = fd_rms.numpy().flatten().tolist()  # [f0, f1, …, f127]

                    for i in range(len(psnr_vals)):
                        example = serialize_psnr(
                            profile.encode('utf-8'), 
                            psnr_vals[i],      
                            tau_rmss[i],        
                            fd_rmss[i],       
                            int(snr_vals[i]),     
                            int(modulation_index) 
                        )
                        writer.write(example)

    print(f"All modulation data saved to {filename}")

def save_val_psnr(dataset, vqvae_model, args):
    len_syms = 960 # (64 (M) * 16 (N)) - (16 (zero_padded) * 4 (column_num)) = 1024 - 64 = 960
    batch_size = args.batch_size
    profile_list = ['NTN-TDL-A', 'NTN-TDL-B', 'NTN-TDL-C', 'NTN-TDL-D']

    num_batches = tf.data.experimental.cardinality(dataset).numpy()
    print(f"Total batches in dataset: {num_batches}")

    filename = os.path.join(args.save_dir, "doppler_psnr_val_dataset.tfrecord")
    if file_exists_nonempty(filename):
        print(f"[SKIP] Found existing val tfrecord: {filename}")
        return

    os.makedirs(args.save_dir, exist_ok=True)
    with tf.io.TFRecordWriter(filename) as writer:
        for profile in profile_list:
            print(f"Processing data for profile: {profile}")
            for mod in range(5):
                seed = tf.constant([123, 42], dtype=tf.int32)
                snr_train = tf.random.stateless_uniform(shape=[num_batches, args.batch_size, 1],
                                                        minval=0, maxval=60,
                                                        seed=seed, dtype=tf.int32)
                ds_iter = iter(dataset.take(num_batches))
                step = 0
                for images in tqdm(ds_iter, total=num_batches, desc=f"Modulation {mod}"):
                    snr_train_val = snr_train[step]
                    step += 1
                    modulation_index = tf.constant(mod, dtype=tf.int32)
                    M_mod = modulation_schemes[int(modulation_index)]['modulation_order']
                    modulate_fn = modulation_schemes[int(modulation_index)]['modulate_fn']
                    demodulate_fn = modulation_schemes[int(modulation_index)]['demodulate_fn']

                    encoded_features, z_shape = vqvae_model.encode(images, modulation_index, training=False)
                    _, code_indices, _ = vqvae_model.quantize(encoded_features, training=False)
                    code_bits = vqvae_model.code2bit(code_indices)
                    code_bits = tf.cast(code_bits, tf.float32)

                    padded_length = tf.case([
                        (tf.equal(modulation_index, 0), lambda: tf.constant(9)),
                        (tf.equal(modulation_index, 1), lambda: tf.constant(10)),
                        (tf.equal(modulation_index, 2), lambda: tf.constant(12)),
                        (tf.equal(modulation_index, 3), lambda: tf.constant(12)),
                    ], default=lambda: tf.constant(16), exclusive=True)

                    group_size = BITS_PER_SYMBOL[M_mod]
                    original_length = tf.shape(code_bits)[-1]
                    remainder = tf.math.floormod(original_length, group_size)
                    pad_size = tf.cond(
                        tf.equal(remainder, 0),
                        lambda: 0,
                        lambda: group_size - remainder
                    )
                    code_bits_padded = tf.pad(code_bits, [[0, 0], [0, pad_size]], constant_values=0)

                    symbols = vqvae_model.modulate(code_bits_padded, M_mod, modulate_fn)
                    symbols = tf.reshape(symbols, [batch_size, -1])
                    sym_len = symbols.shape[1]

                    bits_per_symbol = tf.math.log(tf.cast(M_mod, tf.float32)) / tf.math.log(2.0)
                    num_symbols = 2304 / bits_per_symbol 
                    num_frames = tf.cast(tf.math.ceil(num_symbols / len_syms), tf.int32)
                    total_symbols_needed = num_frames * len_syms
                    pad_len = total_symbols_needed - sym_len

                    padded_symbols = tf.pad(symbols, [[0, 0], [0, pad_len]], "CONSTANT", constant_values=0)
                    frames = tf.reshape(padded_symbols, [batch_size, num_frames, len_syms])
                    snr_per_sample = tf.reshape(snr_train_val, [batch_size])
                    noisy_frames, tau_rms, fd_rms = tf.map_fn(
                        lambda args: tf.map_fn(
                            lambda frame: apply_doppler_channel_tf(
                                frame,
                                args[1],    
                                modulation_index,
                                profile
                            ),
                            args[0],            
                            fn_output_signature=(
                                tf.complex64, tf.float32, tf.float32
                            )
                        ),
                        (frames, snr_per_sample), 
                        fn_output_signature=(
                            tf.complex64, tf.float32, tf.float32
                        )
                    )
                    noisy_symbols = tf.reshape(noisy_frames, [batch_size, -1])[:, :sym_len]

                    noisy_symbols = tf.reshape(noisy_symbols, (-1, ))
                    received_bits = vqvae_model.demodulate(noisy_symbols, M_mod, demodulate_fn)
                    received_bits = tf.cast(received_bits, tf.int32)
                    received_bits = tf.reshape(received_bits, (-1, padded_length))
                    received_bits = received_bits[:, :9]

                    recovered_indices = vqvae_model.bit2code(received_bits)
                    mapping_vector = vqvae_model.vq_layer.embed_code(
                        recovered_indices,
                        [batch_size, 16, 16, args.embedding_dim]
                    )
                    mapping_vector = tf.reshape(mapping_vector, (-1, 256, vqvae_model.embedding_dim))
                    recon = vqvae_model.decode(mapping_vector, modulation_index, [batch_size, 16, 16, 64], training=False)

                    psnr_val = tf.image.psnr(
                        tf.cast(images, tf.float32),
                        tf.cast(recon, tf.float32),
                        max_val=1.0
                    )
                    psnr_vals = psnr_val.numpy().tolist()    # [p0, p1, …, p127]
                    snr_vals  = snr_train_val.numpy().flatten().tolist()  # [s0, s1, …, s127]
                    tau_rmss = tau_rms.numpy().flatten().tolist()  # [t0, t1, …, t127]
                    fd_rmss = fd_rms.numpy().flatten().tolist()  # [f0, f1, …, f127]

                    for i in range(len(psnr_vals)):
                        example = serialize_psnr(
                            profile.encode('utf-8'), 
                            psnr_vals[i],        
                            tau_rmss[i],     
                            fd_rmss[i],        
                            int(snr_vals[i]),    
                            int(modulation_index) 
                        )
                        writer.write(example)
    print(f"All modulation data saved to {filename}")

def main(args):
    img_size = args.img_size
    batch_size = args.batch_size
    vqvae_model_dir = args.vqvae_model_dir
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    train_path = os.path.join(save_dir, "doppler_psnr_train_dataset.tfrecord")
    val_path   = os.path.join(save_dir, "doppler_psnr_val_dataset.tfrecord")

    if file_exists_nonempty(train_path) and file_exists_nonempty(val_path):
        print(f"[SKIP] Both train/val tfrecords already exist in {save_dir}.")
        sys.exit(0)

    if args.dataset_name == 'eurosat':
        train_ds_raw = tfds.load(args.dataset_name, split="train[:80%]", with_info=False, shuffle_files=False)
        val_ds_raw = tfds.load(args.dataset_name, split="train[80%:90%]", with_info=False, shuffle_files=False)
    else:
        train_ds_raw = tfds.load(args.dataset_name, split="train[:90%]", with_info=False, shuffle_files=False)
        val_ds_raw = tfds.load(args.dataset_name, split="train[90%:]", with_info=False, shuffle_files=False)

    train_dataset = (
        train_ds_raw
        .map(lambda x: train_preprocessing(x, img_size), num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(batch_size * 2, seed=42, reshuffle_each_iteration=False)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_dataset = (val_ds_raw
            .map(lambda x: val_preprocessing(x, img_size), num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(batch_size * 2, seed=42, reshuffle_each_iteration=False)
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
            )

    vqvae_model = VQVAE(
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
        num_modulations=args.num_modulations,
        commitment_cost=args.commitment_cost,
        decay=args.decay,
        n_res_block=args.n_res_block,
        img_size=img_size
    )
    vqvae_model = load_models_from_dir(vqvae_model_dir, vqvae_model)

    if not file_exists_nonempty(train_path):
        save_train_psnr(train_dataset, vqvae_model, args)
    else:
        print(f"[SKIP] Train tfrecord already exists: {train_path}")

    if not file_exists_nonempty(val_path):
        save_val_psnr(val_dataset, vqvae_model, args)
    else:
        print(f"[SKIP] Val tfrecord already exists: {val_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='cifar10', help="Name of the dataset to use", choices=['cifar10', 'eurosat'])
    args = parser.parse_args()

    parser.add_argument("--config", type=str, default=f"config/{args.dataset_name}/model_config.yaml", help="Path to the config file")
    args = parser.parse_args()
    config = load_config(args.config)

    parser.add_argument("--img_size", type=int, default=config.get('img_size', 32))
    parser.add_argument("--num_modulations", type=int, default=config.get('num_modulations', 5))
    parser.add_argument("--num_embeddings", type=int, default=config.get('num_embeddings', 512))
    parser.add_argument("--commitment_cost", type=float, default=config.get('commitment_cost', 0.25))
    parser.add_argument("--decay", type=float, default=config.get('decay', 0.99))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--embedding_dim", type=int, default=config.get('embedding_dim', 32))
    parser.add_argument("--n_res_block", type=int, default=config.get('n_res_block', 2))
    parser.add_argument("--vqvae_model_dir", type=str, default=config.get('pretrain_vqvae_model_dir', f'./vqvae_model/{args.dataset_name}'))
    parser.add_argument("--save_dir", type=str, default=config.get('doppler_data_dir', f'./doppler_data/{args.dataset_name}'))

    args = parser.parse_args()

    print(args)
    main(args)
