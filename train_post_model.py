from utils import train_preprocessing, val_preprocessing, load_models_from_dir
from models import VQVAE, PostLMMSENet, VGGFeatureMatchingLoss
import tensorflow as tf
import tensorflow_datasets as tfds
import argparse
from utils import load_config
from modulate_fn_tf import modulate_psk, demodulate_psk, modulate_qam, demodulate_qam
from tqdm import tqdm

import os
vgg_loss = VGGFeatureMatchingLoss()

tf.random.set_seed(42)

def _parse_function(example_proto):
    feature_description = {
        'symbols_real':      tf.io.FixedLenFeature([], tf.string),
        'symbols_imag':      tf.io.FixedLenFeature([], tf.string),
        'noisy_real':        tf.io.FixedLenFeature([], tf.string),
        'noisy_imag':        tf.io.FixedLenFeature([], tf.string),
        'snr':               tf.io.VarLenFeature(tf.int64),
        'modulation_index':  tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)

    symbols_real = tf.io.parse_tensor(parsed['symbols_real'], out_type=tf.float32)
    symbols_imag = tf.io.parse_tensor(parsed['symbols_imag'], out_type=tf.float32)
    noisy_real   = tf.io.parse_tensor(parsed['noisy_real'],   out_type=tf.float32)
    noisy_imag   = tf.io.parse_tensor(parsed['noisy_imag'],   out_type=tf.float32)

    symbols = tf.complex(symbols_real, symbols_imag)
    noisy_symbols = tf.complex(noisy_real, noisy_imag)

    snr = tf.cast(tf.sparse.to_dense(parsed['snr']), tf.int32)
    mod_idx = tf.cast(parsed['modulation_index'], tf.int32)

    return {
        "symbols":       symbols,
        "noisy_symbols": noisy_symbols,
        "snr":           snr,
        "mod_idx":       mod_idx
    }

def load_tf_dataset(filename, batch_size=32):
    ds = tf.data.TFRecordDataset(filename)
    ds = ds.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(tf.data.AUTOTUNE)

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

@tf.function
def train_step(eq_batch, gt_batch, mod_idx_batch, ori_imgs, comp_imgs, post_model, optimizer, max_len=2304):
    prev_len = tf.shape(eq_batch)[1]
    eq_batch = tf.pad(eq_batch, [[0, 0], [0, max_len - prev_len]], constant_values=0)
    with tf.GradientTape() as tape:
        eq_post = post_model(eq_batch, mod_idx_batch, training=True)
        eq_post = eq_post[:, :tf.shape(gt_batch)[1]] 
        recon_loss = tf.reduce_mean((ori_imgs - comp_imgs) ** 2)
        # p_loss = vgg_loss(ori_imgs, comp_imgs) * 0.01
        loss = tf.reduce_mean(tf.abs(eq_post - gt_batch)**2) + recon_loss # MSE
        
    grads = tape.gradient(loss, post_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, post_model.trainable_variables))
    return loss

def reconstruct_images(vqvae_model, eq_post, symbols, images, M_mod, modulation_index, demodulate_fn, padded_length, batch_size):
    eq_post = tf.reshape(eq_post, (-1,))
    received_bits = vqvae_model.demodulate(eq_post, M_mod, demodulate_fn)
    received_bits = tf.cast(received_bits, dtype=tf.int32)
    received_bits = tf.reshape(received_bits, (-1, padded_length))
    received_bits = received_bits[:, :9]

    recovered_indices = vqvae_model.bit2code(received_bits)   

    mapping_vector = vqvae_model.vq_layer.embed_code(
            recovered_indices,
            [symbols.shape[0], 16, 16, args.embedding_dim])
    mapping_vector = tf.reshape(mapping_vector, (-1, 256, vqvae_model.embedding_dim))
    recon = vqvae_model.decode(mapping_vector, modulation_index, [batch_size, 16, 16, 64], training=False)

    psnr_val = tf.image.psnr(images, recon, max_val=1.0)
    psnr_val = tf.reduce_mean(psnr_val)
    return recon, psnr_val

def train_model(dataset, symbol_train_dataset, vqvae_model, post_model, optimizer, args):
    batch_size = args.batch_size

    total_loss = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    ori_psnr_values = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    comp_psnr_values = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    count_batches = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    dataset = tqdm(dataset, desc="Training", total=len(dataset))
    for images, batch in zip(dataset, symbol_train_dataset):
        symbols = batch["symbols"][0]         # shape: (batch_size, L, ...) 
        noisy_symbols = batch["noisy_symbols"][0]       # shape: (batch_size, L, ...)
        modulation_index = batch["mod_idx"][0]          # shape: (batch_size,)
        snr = batch["snr"][0]

        if modulation_index == 0:
            padded_length = 9
        elif modulation_index == 1:
            padded_length = 10
        elif modulation_index == 2:
            padded_length = 12
        elif modulation_index == 3:
            padded_length = 12
        elif modulation_index == 4:
            padded_length = 16

        M_mod = modulation_schemes[modulation_index]['modulation_order']
        demodulate_fn = modulation_schemes[modulation_index]['demodulate_fn']

        modulation_index_batch  = tf.fill([batch_size, 1], value=modulation_index)

        eq_post = post_model(noisy_symbols, modulation_index_batch, training=False)
        eq_post = eq_post[:, :tf.shape(symbols)[1]] 

        ori_imgs, ori_psnr_val = reconstruct_images(
            vqvae_model, symbols, symbols, images, M_mod, modulation_index, demodulate_fn, padded_length, batch_size)
        comp_imgs, comp_psnr_val = reconstruct_images(
            vqvae_model, eq_post, symbols, images, M_mod, modulation_index, demodulate_fn, padded_length, batch_size)
        ori_psnr_values.assign_add(ori_psnr_val)
        comp_psnr_values.assign_add(comp_psnr_val)

        loss = train_step(
            noisy_symbols, symbols, modulation_index_batch, ori_imgs, comp_imgs, post_model, optimizer
        )
        loss = tf.reduce_mean(loss)

        total_loss.assign_add(loss)
        count_batches.assign_add(1.0)


    ori_psnr_values = tf.math.divide_no_nan(ori_psnr_values, count_batches)
    comp_psnr_values = tf.math.divide_no_nan(comp_psnr_values, count_batches)

    mean_loss = tf.math.divide_no_nan(total_loss, count_batches)
    return mean_loss, ori_psnr_values, comp_psnr_values

def validation_model(dataset, symbol_val_dataset, vqvae_model, post_model, args):
    batch_size = args.batch_size

    total_loss = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    ori_psnr_values = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    comp_psnr_values = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    count_batches = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    dataset = tqdm(dataset, desc="Training", total=len(dataset))

    for images, batch in zip(dataset, symbol_val_dataset):
        symbols = batch["symbols"][0]         # shape: (batch_size, L, ...) 
        noisy_symbols = batch["noisy_symbols"][0]       # shape: (batch_size, L, ...)
        modulation_index = batch["mod_idx"][0]          # shape: (batch_size,)
        snr = batch["snr"][0]

        if modulation_index == 0:
            padded_length = 9
        elif modulation_index == 1:
            padded_length = 10
        elif modulation_index == 2:
            padded_length = 12
        elif modulation_index == 3:
            padded_length = 12
        elif modulation_index == 4:
            padded_length = 16

        M_mod = modulation_schemes[modulation_index]['modulation_order']
        demodulate_fn = modulation_schemes[modulation_index]['demodulate_fn']

        modulation_index_batch  = tf.fill([symbols.shape[0], 1], value=modulation_index)

        eq_post = post_model(noisy_symbols, modulation_index_batch, training=False)
        eq_post = eq_post[:, :tf.shape(symbols)[1]] 

        ori_imgs, ori_psnr_val = reconstruct_images(
            vqvae_model, noisy_symbols, symbols, images, M_mod, modulation_index, demodulate_fn, padded_length, batch_size)
        comp_imgs, comp_psnr_val = reconstruct_images(
            vqvae_model, eq_post, symbols, images, M_mod, modulation_index, demodulate_fn, padded_length, batch_size)

        recon_loss = tf.reduce_mean((ori_imgs - comp_imgs) ** 2)
        loss = tf.reduce_mean(tf.abs(eq_post - symbols)**2) + recon_loss # MSE

        total_loss.assign_add(loss)
        count_batches.assign_add(1.0)
        ori_psnr_values.assign_add(ori_psnr_val)
        comp_psnr_values.assign_add(comp_psnr_val)

    ori_psnr_values = tf.math.divide_no_nan(ori_psnr_values, count_batches)
    comp_psnr_values = tf.math.divide_no_nan(comp_psnr_values, count_batches)

    mean_loss = tf.math.divide_no_nan(total_loss, count_batches)
    return mean_loss, ori_psnr_values, comp_psnr_values

def main(args):
    img_size = args.img_size
    batch_size = args.batch_size
    vqvae_model_dir = args.vqvae_model_dir
    max_epoch = args.max_epoch
    post_model_dir = args.post_model_dir

    os.makedirs(post_model_dir, exist_ok=True)

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
        img_size=img_size,
    )
    vqvae_model = load_models_from_dir(vqvae_model_dir, vqvae_model)

    opt = tf.optimizers.Adam(learning_rate=args.learning_rate)
    post_model = PostLMMSENet()

    best_val_loss = -1e3

    for epoch in range(args.max_epoch):
        ds_mod  = load_tf_dataset(f"doppler_data/{args.dataset_name}/doppler_symbol_train_dataset.tfrecord", batch_size=1)
        ds_mod_val  = load_tf_dataset(f"doppler_data/{args.dataset_name}/doppler_symbol_val_dataset.tfrecord", batch_size=1)

        mean_loss, ori_psnr_val, comp_psnr_val = train_model(train_dataset, ds_mod, vqvae_model, post_model, opt, args)
        print(f"Epoch {epoch + 1}/{max_epoch}, Train Loss: {mean_loss.numpy()}, LMMSE PSNR: {ori_psnr_val.numpy()}, Post LMMSE PSNR: {comp_psnr_val.numpy()}")
        mean_val_loss, ori_psnr_validation, comp_psnr_validation = validation_model(val_dataset, ds_mod_val, vqvae_model, post_model, args)
        print(f"Val Loss: {mean_val_loss.numpy()}, LMMSE PSNR: {ori_psnr_validation.numpy()}, Post LMMSE PSNR: {comp_psnr_validation.numpy()}")

        # if (epoch + 1) % 10 == 0:
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            post_model.save_weights(post_model_dir + "/post_model.h5")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='cifar10', help="Name of the dataset to use", choices=['cifar10', 'eurosat'])
    args = parser.parse_args()

    parser.add_argument("--config", type=str, default=f"config/{args.dataset_name}/model_config.yaml", help="Path to the config file")
    args = parser.parse_args()
    config = load_config(args.config)

    parser.add_argument("--img_size", type=int, default=config.get('img_size', 32))
    parser.add_argument("--learning_rate", type=float, default=config.get('post_lr', 1e-4))
    parser.add_argument("--max_epoch", type=int, default=config.get('post_model_max_epoch', 50))
    parser.add_argument("--num_modulations", type=int, default=config.get('num_modulations', 5))
    parser.add_argument("--num_embeddings", type=int, default=config.get('num_embeddings', 512))
    parser.add_argument("--commitment_cost", type=float, default=config.get('commitment_cost', 0.25))
    parser.add_argument("--decay", type=float, default=config.get('decay', 0.99))
    parser.add_argument("--batch_size", type=int, default=config.get('post_batch_size', 32))
    parser.add_argument("--embedding_dim", type=int, default=config.get('embedding_dim', 32))
    parser.add_argument("--n_res_block", type=int, default=config.get('n_res_block', 2))
    parser.add_argument("--vqvae_model_dir", type=str, default=config.get('pretrain_vqvae_model_dir', f'./vqvae_model/{args.dataset_name}'))
    parser.add_argument("--post_model_dir", type=str, default=config.get('post_model_dir', f'post_model/{args.dataset_name}'))
    args = parser.parse_args()

    print(args)
    main(args)
