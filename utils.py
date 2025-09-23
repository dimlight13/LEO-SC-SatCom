import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import yaml

def augment(img):
    return tf.image.random_flip_left_right(img, seed=42)

def resize_and_rescale(img, size, clip_min=0.0, clip_max=1.0):
    height = tf.shape(img)[0]
    width = tf.shape(img)[1]
    crop_size = tf.minimum(height, width)
    img = tf.image.crop_to_bounding_box(
        img,
        (height - crop_size) // 2,
        (width - crop_size) // 2,
        crop_size,
        crop_size,
    )
    img = tf.cast(img, dtype=tf.float32)
    img = tf.image.resize(img, size=size, antialias=True)
    img = img / 255.0
    img = tf.clip_by_value(img, clip_min, clip_max)
    return img

def train_preprocessing(x, img_size):
    img = x["image"]
    img = resize_and_rescale(img, size=(img_size, img_size))
    img = augment(img)
    return img

def val_preprocessing(x, img_size):
    img = x["image"]
    img = resize_and_rescale(img, size=(img_size, img_size))
    return img

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_models_from_dir(save_dir, model):
    dummy_input = tf.zeros((1, 32, 32, 3))
    encoded_features, z_shape = model.encode(dummy_input, 0, training=True)
    quantized, _, _ = model.quantize(encoded_features, training=True)
    _ = model.decode(quantized, 0, z_shape, training=True)

    model.encoder.load_weights(save_dir + '/encoder.h5')
    model.inner_encoder.load_weights(save_dir + '/inner_encoder.h5')
    model.vq_layer.load_weights(save_dir + '/vq_layer.h5')
    model.inner_decoder.load_weights(save_dir + '/inner_decoder.h5')
    model.decoder.load_weights(save_dir + '/decoder.h5')
    return model

def save_images(originals, reconstructions, save_dir, epoch, args, modulation_index):
    fig, axes = plt.subplots(2, args.num_samples, figsize=(args.num_samples * 2, 4))
    for i in range(args.num_samples):
        axes[0, i].imshow(originals[i])
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructions[i])
        axes[1, i].axis('off')
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_dir + '/epoch_{}_mod_{}.png'.format(epoch, modulation_index))
    plt.close()

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
        H = np.ones_like(noise, dtype=np.complex64) 
        
    elif channel_type == 'rayleigh':
        H = (h_real + 1j * h_imag).astype(np.complex64) * np.sqrt(0.5)
        noisy_x = H * inputs + noise
        noisy_x = noisy_x / H 
        
    elif channel_type == 'rician':
        K_factor_dB = 6.0  # K-factor (dB)
        K_factor = 10 ** (K_factor_dB / 10.0)
        h_direct = np.complex64(np.sqrt(K_factor / (K_factor + 1)))
        h_nlos = (h_real + 1j * h_imag).astype(np.complex64)
        H = h_direct + h_nlos
        noisy_x = H * inputs + noise
        noisy_x = noisy_x / H 
    else:
        raise ValueError("Invalid channel type. Choose 'awgn', 'rayleigh', or 'rician'.")

    noisy_x = np.reshape(noisy_x, (-1,), order='F')
    return noisy_x

def channel_effects(inputs, batch_size, snr, channel_type='awgn'):
    inputs = tf.cast(tf.reshape(inputs, [batch_size, -1]), dtype=tf.complex64)

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
