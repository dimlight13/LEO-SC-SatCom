import argparse
import os
import tensorflow as tf
import keras
from keras import optimizers
from tqdm import tqdm
from keras.optimizers.schedules import ExponentialDecay
from modulate_fn_tf import modulate_psk, modulate_qam
from modulate_fn_tf import demodulate_psk, demodulate_qam
from models import VQVAE, VGGFeatureMatchingLoss
from utils import save_images, channel_effects, train_preprocessing, val_preprocessing, load_config
import gc
import tensorflow_datasets as tfds
tf.random.set_seed(42)

MODULATION_ORDERS = [2, 4, 16, 64, 256]

ALPHA_K = {
    2: 3.0,
    4: 1.5,
    16: 1.0,
    64: 0.7,
    256: 0.5
}

BETA_K = {
    k: 0.25 * ALPHA_K[k] for k in MODULATION_ORDERS
}

LAMBDA_K = {
    2: 1.0,
    4: 1.0,
    16: 1.0,
    64: 1.0,
    256: 1.0
}

BITS_PER_SYMBOL = {
    2: 1,   # BPSK
    4: 2,   # QPSK
    16: 4,  # 16-QAM
    64: 6,  # 64-QAM
    256: 8  # 256-QAM
}
FEATURE_DIMENSIONS = {2: 16, 4: 32, 16: 64, 64: 128, 256: 256}  

vgg_loss = VGGFeatureMatchingLoss()

@tf.function
def train_step(x_batch_train, modulation_index, M_mod, snr, modulate_fn, demodulate_fn, model, optimizer, mse_loss_fn, train_loss, alpha_k, beta_k, lambda_k, batch_size):
    with tf.GradientTape() as tape:
        encoded_features, z_shape = model.encode(x_batch_train, modulation_index, training=True)
        quantized, code_indices, flat_inputs = model.quantize(encoded_features, training=True)

        code_bits = model.code2bit(code_indices)
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
        padded_length = tf.shape(code_bits_padded)[-1]
        
        symbols = model.modulate(code_bits_padded , M_mod, modulate_fn)

        noisy_symbols = channel_effects(symbols, batch_size, snr, 'awgn')

        received_bits = model.demodulate(noisy_symbols, M_mod, demodulate_fn)

        received_bits = tf.reshape(received_bits, (-1, padded_length))
        received_bits = received_bits[:, :original_length]

        recovered_indices = model.bit2code(received_bits)
        quantized_recovered = model.vq_layer.embed_code(recovered_indices, [z_shape[0], z_shape[1], z_shape[2], model.embedding_dim])
        quantized = tf.reshape(quantized, (quantized_recovered.shape))

        encodings_recovered = tf.one_hot(tf.reshape(recovered_indices, [-1]), model.num_embeddings)
        flat_inputs = tf.reshape(flat_inputs, [-1, model.embedding_dim])
        model.vq_layer.update_codebook(encodings_recovered, flat_inputs)

        flat_quan = tf.reshape(flat_inputs, (-1, z_shape[1] * z_shape[2], model.embedding_dim))
        quantized = tf.reshape(quantized, (-1, z_shape[1] * z_shape[2], model.embedding_dim))
        quantized_recovered = tf.reshape(quantized_recovered, (-1, z_shape[1] * z_shape[2], model.embedding_dim))

        x_recon = model.decode(quantized, modulation_index, z_shape, training=True)

        codebook_loss = tf.reduce_mean((tf.stop_gradient(quantized_recovered) - flat_quan) ** 2)
        commitment_loss = tf.reduce_mean((quantized_recovered - tf.stop_gradient(flat_quan)) ** 2)
        recon_loss = mse_loss_fn(x_batch_train, x_recon)
        p_loss = vgg_loss(x_batch_train, x_recon)
        loss_k = recon_loss + alpha_k * codebook_loss + beta_k * commitment_loss
        loss = lambda_k * loss_k + p_loss * 0.01

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss.update_state(loss)

@tf.function
def val_step(x_batch_val, modulation_index, M_mod, snr, modulate_fn, demodulate_fn, model, mse_loss_fn, val_loss, batch_size):
    encoded_features, z_shape = model.encode(x_batch_val, modulation_index, training=False)
    quantized, code_indices, flat_inputs = model.quantize(encoded_features, training=False)

    code_bits = model.code2bit(code_indices)
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
    padded_length = tf.shape(code_bits_padded)[-1]
    
    symbols = model.modulate(code_bits_padded, M_mod, modulate_fn)
    noisy_symbols = channel_effects(symbols, batch_size, snr, 'awgn')
    received_bits = model.demodulate(noisy_symbols, M_mod, demodulate_fn)

    received_bits = tf.reshape(received_bits, (-1, padded_length))
    received_bits = received_bits[:, :original_length]

    recovered_indices = model.bit2code(received_bits)
    mapping_vector = model.vq_layer.embed_code(recovered_indices, [z_shape[0], z_shape[1], z_shape[2], model.embedding_dim])
    x_recon = model.decode(mapping_vector, modulation_index, z_shape, training=False)

    recon_loss = mse_loss_fn(x_batch_val, x_recon)
    loss = recon_loss
    val_loss.update_state(loss)

def train(epoch, train_dataset, val_dataset, model, optimizer, args):
    mse_loss_fn = keras.losses.MeanSquaredError()
    train_loss = keras.metrics.Mean()
    val_loss = keras.metrics.Mean()

    snr_boundaries = [0, 5, 12, 20, 26, 30]

    modulation_schemes = [
        {'modulation_order': 2, 'modulate_fn': modulate_psk, 'demodulate_fn': demodulate_psk},
        {'modulation_order': 4, 'modulate_fn': modulate_psk, 'demodulate_fn': demodulate_psk},
        {'modulation_order': 16, 'modulate_fn': modulate_qam, 'demodulate_fn': demodulate_qam},
        {'modulation_order': 64, 'modulate_fn': modulate_qam, 'demodulate_fn': demodulate_qam},
        {'modulation_order': 256, 'modulate_fn': modulate_qam, 'demodulate_fn': demodulate_qam},
    ]

    batch_size = args.batch_size

    for step, x_batch_train in enumerate(tqdm(train_dataset, desc=f'Epoch {epoch+1}/{args.max_epoch} - Training')):
        modulation_index_tf = tf.random.uniform([], minval=0, maxval=args.num_modulations, dtype=tf.int32)
        modulation_index = modulation_index_tf.numpy()
        modulation_scheme = modulation_schemes[modulation_index]
        modulation_order = modulation_scheme['modulation_order']
        modulate_fn = modulation_scheme['modulate_fn']
        demodulate_fn = modulation_scheme['demodulate_fn']

        snr_value_tf = tf.random.uniform([], minval=0.0, maxval=30.0, dtype=tf.float32)
        snr_value = snr_value_tf.numpy()

        modulation_index_tensor = tf.constant(modulation_index, dtype=tf.int32)
        snr_train = tf.fill([args.batch_size, 1], value=snr_value)

        alpha_k = ALPHA_K[modulation_order]
        beta_k = BETA_K[modulation_order]
        lambda_k = LAMBDA_K[modulation_order]

        train_step(x_batch_train, modulation_index_tensor, modulation_order, snr_train, modulate_fn, demodulate_fn, model, optimizer, mse_loss_fn, train_loss, alpha_k, beta_k, lambda_k, batch_size)

    for x_batch_val in val_dataset:
        snr_value_tf_val = tf.random.uniform([], minval=0.0, maxval=30.0, dtype=tf.float32)
        snr_value_val = snr_value_tf_val.numpy()

        modulation_index = None
        for k in range(len(snr_boundaries) - 1):
            if snr_boundaries[k] <= snr_value_val < snr_boundaries[k + 1]:
                modulation_index = k
                break

        snr_eval = tf.fill([args.batch_size, 1], value=snr_value_val)

        modulation_scheme = modulation_schemes[modulation_index]
        modulation_order = modulation_scheme['modulation_order']
        modulate_fn = modulation_scheme['modulate_fn']
        demodulate_fn = modulation_scheme['demodulate_fn']

        val_step(x_batch_val, modulation_index, modulation_order, snr_eval, modulate_fn, demodulate_fn, model, mse_loss_fn, val_loss, batch_size)

    print(f'Epoch {epoch+1}, Train Loss: {train_loss.result().numpy():.4f}, '
          f'Val Loss: {val_loss.result().numpy():.4f}')

    train_loss.reset_states()
    val_loss.reset_states()

def test(test_dataset, model, args, channel_type):
    snr_boundaries = [0, 5, 12, 20, 26, 30]

    modulation_schemes = [
        {'modulation_order': 2, 'modulate_fn': modulate_psk, 'demodulate_fn': demodulate_psk},
        {'modulation_order': 4, 'modulate_fn': modulate_psk, 'demodulate_fn': demodulate_psk},
        {'modulation_order': 16, 'modulate_fn': modulate_qam, 'demodulate_fn': demodulate_qam},
        {'modulation_order': 64, 'modulate_fn': modulate_qam, 'demodulate_fn': demodulate_qam},
        {'modulation_order': 256, 'modulate_fn': modulate_qam, 'demodulate_fn': demodulate_qam},
    ]

    sample_images = next(iter(test_dataset))
    snr_value_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    batch_size = args.batch_size

    for snr_value in snr_value_list:
        modulation_index = None
        for k in range(len(snr_boundaries) - 1):
            if snr_boundaries[k] <= snr_value < snr_boundaries[k + 1]:
                modulation_index = k
                break

        snr_value = tf.cast(snr_value, dtype=tf.float32)
        snr_eval = tf.fill([args.batch_size, 1], value=snr_value)
        modulation_scheme = modulation_schemes[modulation_index]
        modulation_order = modulation_scheme['modulation_order']
        modulate_fn = modulation_scheme['modulate_fn']
        demodulate_fn = modulation_scheme['demodulate_fn']

        modulation_index_tensor = tf.constant(modulation_index, dtype=tf.int32)

        encoded_features, z_shape = model.encode(sample_images, modulation_index_tensor, training=False)
        quantized, code_indices, flat_inputs = model.quantize(encoded_features, training=False)

        code_bits = model.code2bit(code_indices)
        code_bits = tf.cast(code_bits, tf.float32)

        group_size = BITS_PER_SYMBOL[modulation_order]
        original_length = tf.shape(code_bits)[-1]
        remainder = tf.math.floormod(original_length, group_size)
        pad_size = tf.cond(
            tf.equal(remainder, 0),
            lambda: 0,
            lambda: group_size - remainder
        )
        code_bits_padded = tf.pad(code_bits, [[0, 0], [0, pad_size]], constant_values=0)
        padded_length = tf.shape(code_bits_padded)[-1]

        symbols = model.modulate(code_bits_padded, modulation_order, modulate_fn)
        noisy_symbols = channel_effects(symbols, batch_size, snr_eval, channel_type)
        received_bits = model.demodulate(noisy_symbols, modulation_order, demodulate_fn)

        received_bits = tf.reshape(received_bits, (-1, padded_length))
        received_bits = received_bits[:, :original_length]

        recovered_indices = model.bit2code(received_bits)
        mapping_vector = model.vq_layer.embed_code(recovered_indices, [z_shape[0], z_shape[1], z_shape[2], model.embedding_dim])
        reconstructions = model.decode(mapping_vector, modulation_index, z_shape, training=False)

        psnr_values = tf.image.psnr(sample_images, reconstructions, max_val=1.0)
        mean_psnr = tf.reduce_mean(psnr_values)
        print(f'Channel {channel_type}, Modulation {modulation_index}, Mean PSNR: {mean_psnr.numpy():.2f} dB')

def main(args):
    img_size = args.img_size
    batch_size = args.batch_size

    save_model_dir = args.save_model_dir
    save_img_dir = args.save_img_dir
    os.makedirs(save_model_dir, exist_ok=True)
    os.makedirs(save_img_dir, exist_ok=True)

    if args.dataset_name == 'eurosat':
        train_ds_raw = tfds.load(args.dataset_name, split="train[:80%]", with_info=False, shuffle_files=False)
        val_ds_raw = tfds.load(args.dataset_name, split="train[80%:90%]", with_info=False, shuffle_files=False)
        test_ds_raw = tfds.load(args.dataset_name, split="train[90%:]", with_info=False, shuffle_files=True)
    else:
        train_ds_raw = tfds.load(args.dataset_name, split="train[:90%]", with_info=False, shuffle_files=False)
        val_ds_raw = tfds.load(args.dataset_name, split="train[90%:]", with_info=False, shuffle_files=False)
        test_ds_raw = tfds.load(args.dataset_name, split="test", with_info=False, shuffle_files=True)

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

    test_dataset = (test_ds_raw
            .map(lambda x: val_preprocessing(x, img_size), num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(batch_size * 2, seed=42, reshuffle_each_iteration=False)
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
            )

    num_modulations = args.num_modulations

    model = VQVAE(num_embeddings=args.num_embeddings,
                  embedding_dim=args.embedding_dim,
                  num_modulations=num_modulations,
                  commitment_cost=args.commitment_cost,
                  decay=args.decay,
                  n_res_block=args.n_res_block,
                  img_size=img_size)

    steps_per_epoch = len(train_dataset) 
    decay_steps = steps_per_epoch * 20 
    learning_rate_schedule = ExponentialDecay(
        initial_learning_rate=args.init_lr, 
        decay_steps=decay_steps,
        decay_rate=0.5,
        staircase=True
    )

    optimizer = optimizers.Adam(learning_rate=learning_rate_schedule)
    snr_boundaries = [0, 5, 12, 20, 26, 30]

    modulation_schemes = [
        {'modulation_order': 2, 'modulate_fn': modulate_psk, 'demodulate_fn': demodulate_psk},
        {'modulation_order': 4, 'modulate_fn': modulate_psk, 'demodulate_fn': demodulate_psk},
        {'modulation_order': 16, 'modulate_fn': modulate_qam, 'demodulate_fn': demodulate_qam},
        {'modulation_order': 64, 'modulate_fn': modulate_qam, 'demodulate_fn': demodulate_qam},
        {'modulation_order': 256, 'modulate_fn': modulate_qam, 'demodulate_fn': demodulate_qam},
    ]

    batch_size = args.batch_size
    for epoch in range(args.max_epoch):
        train(epoch, train_dataset, val_dataset, model, optimizer, args)

        tf.keras.backend.clear_session()
        gc.collect()

        if (epoch + 1) % args.save_interval == 0:
            sample_images = next(iter(val_dataset))
            snr_value_list = [3, 10, 15, 23, 28]

            for snr_value in snr_value_list:
                modulation_index = None
                for k in range(len(snr_boundaries) - 1):
                    if snr_boundaries[k] <= snr_value < snr_boundaries[k + 1]:
                        modulation_index = k
                        break

                snr_value = tf.cast(snr_value, dtype=tf.float32)
                snr_eval = tf.fill([args.batch_size, 1], value=snr_value)
                modulation_scheme = modulation_schemes[modulation_index]
                modulation_order = modulation_scheme['modulation_order']
                modulate_fn = modulation_scheme['modulate_fn']
                demodulate_fn = modulation_scheme['demodulate_fn']

                modulation_index_tensor = tf.constant(modulation_index, dtype=tf.int32)
                
                encoded_features, z_shape = model.encode(sample_images, modulation_index, training=False)
                quantized, code_indices, flat_inputs = model.quantize(encoded_features, training=False)
                
                code_bits = model.code2bit(code_indices)
                code_bits = tf.cast(code_bits, tf.float32)

                group_size = BITS_PER_SYMBOL[modulation_order] 
                original_length = tf.shape(code_bits)[-1]
                remainder = tf.math.floormod(original_length, group_size)
                pad_size = tf.cond(
                    tf.equal(remainder, 0),
                    lambda: 0,
                    lambda: group_size - remainder
                )
                code_bits_padded = tf.pad(code_bits, [[0, 0], [0, pad_size]], constant_values=0)
                padded_length = tf.shape(code_bits_padded)[-1]

                symbols = model.modulate(code_bits_padded, modulation_order, modulate_fn)
                noisy_symbols = channel_effects(symbols, batch_size, snr_eval, 'awgn')
                received_bits = model.demodulate(noisy_symbols, modulation_order, demodulate_fn)

                received_bits = tf.reshape(received_bits, (-1, padded_length))
                received_bits = received_bits[:, :original_length]

                recovered_indices = model.bit2code(received_bits)
                mapping_vector = model.vq_layer.embed_code(recovered_indices, [z_shape[0], z_shape[1], z_shape[2], model.embedding_dim])
                reconstructions = model.decode(mapping_vector, modulation_index, z_shape, training=False)

                psnr_values = tf.image.psnr(sample_images, reconstructions, max_val=1.0)
                mean_psnr = tf.reduce_mean(psnr_values)
                print(f'Epoch {epoch+1}, Modulation {modulation_index}, Mean PSNR: {mean_psnr.numpy():.2f} dB')

                save_images(sample_images.numpy(), reconstructions.numpy(), save_img_dir, epoch+1, args, modulation_index)

        if (epoch + 1) % args.save_interval == 0:
            os.makedirs(save_model_dir, exist_ok=True)
            model.encoder.save_weights(save_model_dir + '/encoder.h5')
            model.inner_encoder.save_weights(save_model_dir + '/inner_encoder.h5')
            model.vq_layer.save_weights(save_model_dir + '/vq_layer.h5')
            model.inner_decoder.save_weights(save_model_dir + '/inner_decoder.h5')
            model.decoder.save_weights(save_model_dir + '/decoder.h5')

    channel_type_list = ['awgn', 'rician', 'rayleigh']
    for channel_type in channel_type_list:
        test(test_dataset, model, args, channel_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='cifar10', help="Name of the dataset to use", choices=['cifar10', 'eurosat'])
    args = parser.parse_args()

    parser.add_argument("--config", type=str, default=f"config/{args.dataset_name}/model_config.yaml", help="Path to the config file")
    args = parser.parse_args()
    config = load_config(args.config)

    parser.add_argument("--img_size", type=int, default=config.get('img_size', 32))
    parser.add_argument("--max_epoch", type=int, default=config.get('vqvae_max_epoch', 100))
    parser.add_argument("--num_modulations", type=int, default=config.get('num_modulations', 5))
    parser.add_argument("--num_embeddings", type=int, default=config.get('num_embeddings', 512))
    parser.add_argument("--save_interval", type=int, default=config.get('save_interval', 5))
    parser.add_argument("--commitment_cost", type=float, default=config.get('commitment_cost', 0.25))
    parser.add_argument("--decay", type=float, default=config.get('decay', 0.99))
    parser.add_argument("--init_lr", type=float, default=config.get('vqvae_lr', 1e-3))
    parser.add_argument("--batch_size", type=int, default=config.get('batch_size', 128))
    parser.add_argument("--embedding_dim", type=int, default=config.get('embedding_dim', 32))
    parser.add_argument("--n_res_block", type=int, default=config.get('n_res_block', 2))
    parser.add_argument("--num_samples", type=int, default=config.get('num_samples', 4))
    parser.add_argument("--save_model_dir", type=str, default=config.get('pretrain_vqvae_model_dir', f'./vqvae_model/{args.dataset_name}'), help="Directory to save trained model")
    parser.add_argument("--save_img_dir", type=str, default=config.get('pretrain_vqvae_img_dir', f'./vqvae_images/{args.dataset_name}'), help="Directory to save diffused images")

    args = parser.parse_args()

    print(args)
    main(args)
