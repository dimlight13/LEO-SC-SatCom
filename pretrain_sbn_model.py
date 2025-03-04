import argparse
import os
import tensorflow as tf
import keras
from keras import optimizers
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.optimizers.schedules import ExponentialDecay
from modulate_fn_tf import modulate_psk, modulate_qam
from modulate_fn_tf import demodulate_psk, demodulate_qam
from models import VQVAE
from utils import channel_effects, augment, preprocess_data, save_images

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

@tf.function
def train_step(x_batch_train, modulation_index, M_mod, snr, modulate_fn, demodulate_fn, model, optimizer, mse_loss_fn, train_loss, alpha_k, beta_k, lambda_k, batch_size):
    with tf.GradientTape() as tape:
        quantized, z_shape, code_indices, flat_inputs = model.encode(x_batch_train, modulation_index, training=True)

        code_bits = model.code2bit(code_indices)
        code_bits = tf.cast(code_bits, tf.float32)
        symbols = model.modulate(code_bits, M_mod, modulate_fn)

        noisy_symbols = channel_effects(symbols, batch_size, snr, 'awgn')

        received_bits = model.demodulate(noisy_symbols, M_mod, demodulate_fn)
        recovered_indices = model.bit2code(received_bits)
        quantized_recovered = model.vq_layer.embed_code(recovered_indices, [z_shape[0], z_shape[1], z_shape[2], z_shape[3]])
        quantized = tf.reshape(quantized, (quantized_recovered.shape))

        encodings_recovered = tf.one_hot(tf.reshape(recovered_indices, [-1]), model.num_embeddings)
        flat_inputs = tf.reshape(flat_inputs, [-1, model.embedding_dim])
        model.vq_layer.update_codebook(encodings_recovered, flat_inputs)

        flat_quan = tf.reshape(flat_inputs, (-1, 256, 64))
        quantized = tf.reshape(quantized, (-1, 256, 64))
        quantized_recovered = tf.reshape(quantized_recovered, (-1, 256, 64))

        x_recon = model.decode(quantized, modulation_index, z_shape, training=True)

        codebook_loss = tf.reduce_mean((tf.stop_gradient(quantized_recovered) - flat_quan) ** 2)
        commitment_loss = tf.reduce_mean((quantized_recovered - tf.stop_gradient(flat_quan)) ** 2)
        recon_loss = mse_loss_fn(x_batch_train, x_recon)
        loss_k = recon_loss + alpha_k * codebook_loss + beta_k * commitment_loss
        loss = lambda_k * loss_k

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss.update_state(loss)

def val_step(x_batch_val, modulation_index, M_mod, snr, modulate_fn, demodulate_fn, model, mse_loss_fn, val_loss, batch_size):
    quantized, z_shape, code_indices, flat_inputs  = model.encode(x_batch_val, modulation_index, training=False)

    code_bits = model.code2bit(code_indices)
    code_bits = tf.cast(code_bits, tf.float32)
    symbols = model.modulate(code_bits, M_mod, modulate_fn)
    noisy_symbols = channel_effects(symbols, batch_size, snr, 'awgn')
    received_bits = model.demodulate(noisy_symbols, M_mod, demodulate_fn)
    recovered_indices = model.bit2code(received_bits)
    mapping_vector = model.vq_layer.embed_code(recovered_indices, [z_shape[0], z_shape[1], z_shape[2], z_shape[3]])
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

    modulation_index_tf = tf.random.uniform([], minval=0, maxval=5, dtype=tf.int32)
    modulation_index = modulation_index_tf.numpy()
    modulation_scheme = modulation_schemes[modulation_index]
    modulation_order = modulation_scheme['modulation_order']
    modulate_fn = modulation_scheme['modulate_fn']
    demodulate_fn = modulation_scheme['demodulate_fn']

    for step, x_batch_train in enumerate(tqdm(train_dataset, desc=f'Epoch {epoch+1}/{args.epoch} - Training')):
        snr_value_tf = tf.random.uniform([], minval=0.0, maxval=30.0, dtype=tf.float32)
        snr_value = snr_value_tf.numpy()

        # for k in range(len(snr_boundaries) - 1):
        #     if snr_boundaries[k] <= snr_value < snr_boundaries[k + 1]:
        #         modulation_index = k
        #         break

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
        quantized, z_shape, code_indices, flat_inputs = model.encode(sample_images, modulation_index_tensor, training=False)
        code_bits = model.code2bit(code_indices)
        code_bits = tf.cast(code_bits, tf.float32)
        symbols = model.modulate(code_bits, modulation_order, modulate_fn)
        noisy_symbols = channel_effects(symbols, batch_size, snr_eval, channel_type)
        received_bits = model.demodulate(noisy_symbols, modulation_order, demodulate_fn)
        recovered_indices = model.bit2code(received_bits)
        mapping_vector = model.vq_layer.embed_code(recovered_indices, [z_shape[0], z_shape[1], z_shape[2], z_shape[3]])
        reconstructions = model.decode(mapping_vector, modulation_index, z_shape, training=False)

        psnr_values = tf.image.psnr(sample_images, reconstructions, max_val=1.0)
        mean_psnr = tf.reduce_mean(psnr_values)
        print(f'Channel {channel_type}, Modulation {modulation_index}, Mean PSNR: {mean_psnr.numpy():.2f} dB')

def main(args):
    (x_train, _), (x_test, _) = keras.datasets.cifar10.load_data()
    x_train = preprocess_data(x_train)
    x_test = preprocess_data(x_test)

    x_train, x_val = train_test_split(
        x_train, 
        test_size=0.1, 
        shuffle=True, 
        random_state=42
    )

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train))\
        .shuffle(len(x_train))\
        .map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .batch(args.batch_size, drop_remainder=True)\
        .prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((x_val))\
        .batch(args.batch_size, drop_remainder=True)\
        .prefetch(tf.data.experimental.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test))\
        .batch(args.batch_size, drop_remainder=True)\
        .prefetch(tf.data.experimental.AUTOTUNE)

    num_modulations = len(MODULATION_ORDERS)

    model = VQVAE(num_embeddings=args.num_embeddings,
                  embedding_dim=args.embedding_dim,
                  num_modulations=num_modulations,
                  commitment_cost=args.commitment_cost,
                  decay=args.decay,
                  n_res_block=args.n_res_block)

    steps_per_epoch = len(train_dataset)  # 한 에포크당 스텝 수 계산
    decay_steps = steps_per_epoch * 20  # 20 에포크마다 학습률 감소
    learning_rate_schedule = ExponentialDecay(
        initial_learning_rate=args.lr_max,  # 초기 학습률 0.001
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
    for epoch in range(args.epoch):
        train(epoch, train_dataset, val_dataset, model, optimizer, args)

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
                quantized, z_shape, code_indices, flat_inputs = model.encode(sample_images, modulation_index_tensor, training=False)
                code_bits = model.code2bit(code_indices)
                code_bits = tf.cast(code_bits, tf.float32)
                symbols = model.modulate(code_bits, modulation_order, modulate_fn)
                noisy_symbols = channel_effects(symbols, batch_size, snr_eval, 'awgn')
                received_bits = model.demodulate(noisy_symbols, modulation_order, demodulate_fn)
                recovered_indices = model.bit2code(received_bits)
                mapping_vector = model.vq_layer.embed_code(recovered_indices, [z_shape[0], z_shape[1], z_shape[2], z_shape[3]])
                reconstructions = model.decode(mapping_vector, modulation_index, z_shape, training=False)

                psnr_values = tf.image.psnr(sample_images, reconstructions, max_val=1.0)
                mean_psnr = tf.reduce_mean(psnr_values)
                print(f'Epoch {epoch+1}, Modulation {modulation_index}, Mean PSNR: {mean_psnr.numpy():.2f} dB')

                save_images(sample_images.numpy(), reconstructions.numpy(), epoch+1, args, modulation_index)

        if (epoch + 1) % args.save_interval == 0:
            os.makedirs('checkpoint', exist_ok=True)
            model.encoder.save_weights('checkpoint/pretrain/encoder.h5')
            model.inner_encoder.save_weights('checkpoint/pretrain/inner_encoder.h5')
            model.vq_layer.save_weights('checkpoint/pretrain/vq_layer.h5')
            model.inner_decoder.save_weights('checkpoint/pretrain/inner_decoder.h5')
            model.decoder.save_weights('checkpoint/pretrain/decoder.h5')
            model.save_weights('checkpoint/vqvae.h5')

    channel_type_list = ['awgn', 'rician', 'rayleigh']
    for channel_type in channel_type_list:
        test(test_dataset, model, args, channel_type)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=400)
    parser.add_argument("--lr_max", type=float, default=1e-3)
    parser.add_argument("--lr_min", type=float, default=5e-5)
    parser.add_argument("--cycle_length", type=int, default=10)  # Number of epochs per cycle
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_embeddings", type=int, default=512)
    parser.add_argument("--commitment_cost", type=float, default=0.25)
    parser.add_argument("--decay", type=float, default=0.99)
    parser.add_argument("--n_res_block", type=int, default=2)
    parser.add_argument("--n_res_channel", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--snr_db", type=float, default=10.0)
    args = parser.parse_args()

    print(args)
    main(args)
