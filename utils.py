import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np

def load_models_from_dir(model):
    dummy_input = tf.zeros((1, 32, 32, 3))
    quan, z_shape, _, _ = model.encode(dummy_input, 0, training=True)
    _ = model.decode(quan, 0, z_shape, training=True)

    model.encoder.load_weights('checkpoint/pretrain/encoder.h5')
    model.inner_encoder.load_weights('checkpoint/pretrain/inner_encoder.h5')
    model.vq_layer.load_weights('checkpoint/pretrain/vq_layer.h5')
    model.inner_decoder.load_weights('checkpoint/pretrain/inner_decoder.h5')
    model.decoder.load_weights('checkpoint/pretrain/decoder.h5')
    return model

def save_images(originals, reconstructions, epoch, args, modulation_index):
    fig, axes = plt.subplots(2, args.num_samples, figsize=(args.num_samples * 2, 4))
    for i in range(args.num_samples):
        axes[0, i].imshow(originals[i])
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructions[i])
        axes[1, i].axis('off')
    plt.tight_layout()
    os.makedirs('samples', exist_ok=True)
    plt.savefig(f'samples/epoch_{epoch}_mod_{modulation_index}.png')
    plt.close()

def augment(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image

def preprocess_data(X):
    X = X.astype('float32')
    X = X / 255.0
    return X

def channel_effects_np(inputs, sigma_2, channel_type='awgn'):
    inputs = np.array(np.reshape(inputs, (1, -1)), dtype=np.complex64, order='F')
    std_value = np.sqrt(sigma_2 / 2).astype(np.float32)
    
    if np.iscomplexobj(inputs):
        noise_real = np.random.normal(loc=0.0, scale=std_value, size=inputs.shape)
        noise_imag = np.random.normal(loc=0.0, scale=std_value, size=inputs.shape)
        noise = noise_real + 1j * noise_imag
    else:
        noise = np.random.normal(loc=0.0, scale=std_value, size=inputs.shape)
    
    h_real = np.random.normal(loc=0.0, scale=1.0, size=inputs.shape).astype(np.float32)
    h_imag = np.random.normal(loc=0.0, scale=1.0, size=inputs.shape).astype(np.float32)
    
    if channel_type == 'awgn':
        noisy_x = inputs + noise
        H = np.ones_like(noise, dtype=np.complex64)  # 채널 이득은 1
        
    elif channel_type == 'rayleigh':
        H = (h_real + 1j * h_imag).astype(np.complex64) * np.sqrt(0.5)
        noisy_x = H * inputs + noise
        noisy_x = noisy_x / H  # 채널 보상
        
    elif channel_type == 'rician':
        K_factor_dB = 6.0  # K-factor (dB)
        K_factor = 10 ** (K_factor_dB / 10.0)
        h_direct = np.complex64(np.sqrt(K_factor / (K_factor + 1)))
        h_nlos = (h_real + 1j * h_imag).astype(np.complex64)
        H = h_direct + h_nlos
        noisy_x = H * inputs + noise
        noisy_x = noisy_x / H  # 채널 보상
    else:
        raise ValueError("Invalid channel type. Choose 'awgn', 'rayleigh', or 'rician'.")

    noisy_x = np.reshape(noisy_x, (-1,), order='F')
    return noisy_x

def channel_effects(inputs, snr, channel_type='awgn'):
    inputs = tf.cast(tf.reshape(inputs, [1, -1]), dtype=tf.complex64)

    snr_linear = tf.pow(10.0, snr / 10.0)
    signal_power = tf.reduce_mean(tf.abs(inputs) ** 2, axis=1, keepdims=True)
    noise_power = signal_power / snr_linear
    std_value = tf.cast(tf.sqrt(noise_power / 2), dtype=tf.float32)

    if inputs.dtype.is_complex:
        noise_real = tf.random.normal(tf.shape(inputs), mean=0.0, stddev=std_value)
        noise_imag = tf.random.normal(tf.shape(inputs), mean=0.0, stddev=std_value)
        noise = tf.complex(noise_real, noise_imag)
    else:
        noise = tf.random.normal(tf.shape(inputs), mean=0.0, stddev=std_value)

    h_real = tf.random.normal(tf.shape(inputs), dtype=tf.float32)
    h_imag = tf.random.normal(tf.shape(inputs), dtype=tf.float32)

    if channel_type == 'awgn':
        noisy_x = inputs + noise
        H = tf.ones_like(noise)  # Initialize H as ones

    elif channel_type == 'rayleigh':
        H = tf.complex(h_real, h_imag) * tf.cast(tf.sqrt(0.5), tf.complex64)  # normalize
        noisy_x = H * inputs + noise
        noisy_x /= H
        
    elif channel_type == 'rician':
        K_factor_dB = 6.0 # Default value
        K_factor = 10 ** (K_factor_dB / 10)

        h_direct = tf.complex(tf.sqrt(K_factor / (K_factor + 1)), 0.0)  # LOS component
        h_nlos = tf.complex(h_real, h_imag)
        H = h_direct + h_nlos
        noisy_x = H * inputs + noise
        noisy_x /= H
    else:
        raise ValueError("Invalid channel type. Choose 'awgn', 'rayleigh', or 'rician'.")

    noisy_x = tf.reshape(noisy_x, [-1, ])
    return noisy_x
