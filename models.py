import tensorflow as tf
from keras import layers, Model

class VQVAE(Model):
    def __init__(self, num_embeddings, embedding_dim, num_modulations, 
                 commitment_cost=0.25, decay=0.99, n_res_block=2, name='vqvae', **kwargs):
        super(VQVAE, self).__init__(name=name, **kwargs)

        MODULATION_ORDERS = [2, 4, 16, 64, 256]
        BITS_PER_SYMBOL = {
            2: 1,   # BPSK
            4: 2,   # QPSK
            16: 4,  # 16-QAM
            64: 6,  # 64-QAM
            256: 8  # 256-QAM
        }

        self.num_modulations = num_modulations
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.modulation_orders = tf.constant(MODULATION_ORDERS, dtype=tf.int32)  # [2, 4, 16]
        self.BITS_PER_SYMBOL = tf.constant([BITS_PER_SYMBOL[m] for m in MODULATION_ORDERS], dtype=tf.int32)  # [1, 2, 4]

        channels = [64, 128, 64]

        # Define encoder, inner encoder, single vector quantizer, inner decoder, and decoder
        self.encoder = Encoder(channels, n_res_block, num_modulations)
        self.inner_encoder = InnerEncoder(channels[-1], embedding_dim, num_modulations)
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost, decay)
        self.inner_decoder = InnerDecoder(embedding_dim, channels[-1], num_modulations)
        self.decoder = Decoder(channels[::-1], n_res_block, num_modulations)

    def code2bit(self, code_indices):
        num_bits_per_index = tf.cast(tf.math.ceil(tf.math.log(tf.cast(self.num_embeddings, tf.float32)) / tf.math.log(2.0)), tf.int32)
        code_indices_int32 = tf.cast(code_indices, tf.int32)
        bit_shifts = tf.range(num_bits_per_index, dtype=tf.int32)
        code_bits = tf.reshape(tf.bitwise.right_shift(tf.expand_dims(code_indices_int32, 1), bit_shifts) & 1,
                               [-1, num_bits_per_index])
        return code_bits

    def bit2code(self, bits):
        bits = tf.cast(bits, dtype=tf.int32)
        num_bits_per_index = tf.cast(tf.math.ceil(tf.math.log(tf.cast(self.num_embeddings, tf.float32)) / tf.math.log(2.0)), tf.int32)
        bits_reshaped = tf.reshape(bits, [-1, num_bits_per_index])
        indices = tf.reduce_sum(bits_reshaped * (2 ** tf.range(num_bits_per_index, dtype=tf.int32)), axis=-1)
        return tf.cast(indices, dtype=tf.int32)

    def modulate(self, code_bits, M_mod, modulate_fn):
        symbols = modulate_fn(code_bits, M_mod)
        return symbols

    def demodulate(self, symbols, M_mod, demodulate_fn):
        bits = demodulate_fn(symbols, M_mod)
        return bits

    def encode(self, x, modulation_index, training=False):
        modulation_index = tf.cast(modulation_index, tf.int32)
        z = self.encoder(x, modulation_index, training=training)
        z_ = self.inner_encoder(z, modulation_index, training=training)

        quantized, encoding_indices, flat_inputs = self.vq_layer(z_, training=training)
        return quantized, z.shape, encoding_indices, flat_inputs

    def decode(self, recovered_bits, modulation_index, z_shape, training=False):
        decoding_vector = tf.reshape(recovered_bits, (z_shape[0], z_shape[1] * z_shape[2], z_shape[3]))
        decoding_vector = self.inner_decoder(decoding_vector, modulation_index, 
                                            output_shape=(z_shape[0], z_shape[1], z_shape[2], z_shape[3]), training=training)

        x_recon = self.decoder(decoding_vector, modulation_index, training=training)
        return x_recon

    def call(self, x, modulation_index, training=False):
        symbols, vq_loss, z_shape = self.encode(x, modulation_index, training=training)
        return symbols, vq_loss

class SwitchableBatchNormalization(layers.Layer):
    def __init__(self, num_features, num_modulations, momentum=0.99, epsilon=1e-3, name=None, **kwargs):
        super(SwitchableBatchNormalization, self).__init__(name=name, **kwargs)
        self.num_features = num_features
        self.num_modulations = num_modulations
        self.momentum = momentum
        self.epsilon = epsilon

        self.gamma = self.add_weight(
            shape=(num_modulations, num_features),
            initializer='ones',
            trainable=True,
            name=f'{self.name}_gamma'
        )
        self.beta = self.add_weight(
            shape=(num_modulations, num_features),
            initializer='zeros',
            trainable=True,
            name=f'{self.name}_beta'
        )
        self.moving_mean = self.add_weight(
            shape=(num_modulations, num_features),
            initializer='zeros',
            trainable=False,
            name=f'{self.name}_moving_mean'
        )
        self.moving_variance = self.add_weight(
            shape=(num_modulations, num_features),
            initializer='ones',
            trainable=False,
            name=f'{self.name}_moving_variance'
        )

    def call(self, inputs, modulation_index, training=True):
        modulation_index = tf.cast(modulation_index, tf.int32)
        gamma = self.gamma[modulation_index]  # Shape: (num_features,)
        beta = self.beta[modulation_index]    # Shape: (num_features,)
        moving_mean = self.moving_mean[modulation_index]  # Shape: (num_features,)
        moving_variance = self.moving_variance[modulation_index]  # Shape: (num_features,)

        # Reshape for broadcasting
        gamma = tf.reshape(gamma, [1, 1, 1, -1])
        beta = tf.reshape(beta, [1, 1, 1, -1])

        if training:
            # Compute batch statistics
            batch_mean, batch_variance = tf.nn.moments(inputs, axes=[0, 1, 2], keepdims=False)
            # Update moving averages
            self.moving_mean[modulation_index].assign(
                self.momentum * moving_mean + (1 - self.momentum) * batch_mean)
            self.moving_variance[modulation_index].assign(
                self.momentum * moving_variance + (1 - self.momentum) * batch_variance)
            mean = batch_mean
            variance = batch_variance
        else:
            mean = moving_mean
            variance = moving_variance

        # Reshape mean and variance for broadcasting
        mean = tf.reshape(mean, [1, 1, 1, -1])
        variance = tf.reshape(variance, [1, 1, 1, -1])

        outputs = tf.nn.batch_normalization(
            inputs, mean, variance, beta, gamma, self.epsilon)
        return outputs

class ResBlock(layers.Layer):
    def __init__(self, channels, num_modulations, name=None, **kwargs):
        super(ResBlock, self).__init__(name=name, **kwargs)
        self.conv1 = layers.Conv2D(
            filters=channels,
            kernel_size=3,
            padding='same',
            activation=None,
            name=f'{self.name}_conv1'
        )
        self.sbn1 = SwitchableBatchNormalization(channels, num_modulations, name=f'{self.name}_sbn1')
        self.relu = layers.ReLU()
        self.conv2 = layers.Conv2D(
            filters=channels,
            kernel_size=1,
            padding='same',
            activation=None,
            name=f'{self.name}_conv2'
        )
        self.sbn2 = SwitchableBatchNormalization(channels, num_modulations, name=f'{self.name}_sbn2')

    def call(self, inputs, modulation_index, training=True):
        residual = inputs
        x = self.conv1(inputs)
        x = self.sbn1(x, modulation_index, training=training)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sbn2(x, modulation_index, training=training)
        x = layers.add([x, residual])
        x = self.relu(x)
        return x

class Encoder(Model):
    def __init__(self, channels, n_res_block, num_modulations, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.num_modulations = num_modulations

        self.initial_layers = [
            layers.Conv2D(filters=channels[0], kernel_size=5, strides=1, padding='same', activation=None, name=f'{self.name}_conv1'),
            SwitchableBatchNormalization(channels[0], num_modulations, name=f'{self.name}_sbn1'),
            layers.ReLU()
        ]

        self.downsample_layers = [
            layers.Conv2D(filters=channels[1], kernel_size=5, strides=1, padding='same', activation=None, name=f'{self.name}_conv2'),
            SwitchableBatchNormalization(channels[1], num_modulations, name=f'{self.name}_sbn2'),
            layers.ReLU()
        ]

        self.res_blocks = [ResBlock(channels[1], num_modulations, name=f'{self.name}_resblock_{i}') for i in range(n_res_block)]
        self.final_layers = [
            layers.Conv2D(filters=channels[2], kernel_size=2, strides=2, padding='valid', activation=None, name=f'{self.name}_conv3'),
            SwitchableBatchNormalization(channels[2], num_modulations, name=f'{self.name}_sbn3'),
            layers.ReLU()
        ]

    def call(self, inputs, modulation_index, training=True):
        x = inputs
        for layer in self.initial_layers:
            if isinstance(layer, SwitchableBatchNormalization):
                x = layer(x, modulation_index, training=training)
            else:
                x = layer(x)
        for layer in self.downsample_layers:
            if isinstance(layer, SwitchableBatchNormalization):
                x = layer(x, modulation_index, training=training)
            else:
                x = layer(x)
        for res_block in self.res_blocks:
            x = res_block(x, modulation_index, training=training)
        for layer in self.final_layers:
            if isinstance(layer, SwitchableBatchNormalization):
                x = layer(x, modulation_index, training=training)
            else:
                x = layer(x)
        return x

class Decoder(Model):
    def __init__(self, channels, n_res_block, num_modulations, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.num_modulations = num_modulations

        self.initial_layers = [
            layers.Conv2DTranspose(filters=channels[0], kernel_size=3, strides=1, padding='same', activation=None, name=f'{self.name}_deconv1'),
            SwitchableBatchNormalization(channels[0], num_modulations, name=f'{self.name}_sbn1'),
            layers.ReLU()
        ]

        self.res_blocks = [ResBlock(channels[0], num_modulations, name=f'{self.name}_resblock_{i}') for i in range(n_res_block)]
        self.upsample_layers = [
            layers.Conv2DTranspose(filters=channels[1], kernel_size=2, strides=2, padding='valid', activation=None, name=f'{self.name}_deconv2'),
            SwitchableBatchNormalization(channels[1], num_modulations, name=f'{self.name}_sbn2'),
            layers.ReLU()
        ]

        self.final_layers = [
            layers.Conv2DTranspose(filters=channels[2], kernel_size=5, strides=1, padding='same', activation=None, name=f'{self.name}_deconv3'),
            SwitchableBatchNormalization(channels[2], num_modulations, name=f'{self.name}_sbn3'),
            layers.ReLU(),
            layers.Conv2DTranspose(filters=3, kernel_size=5, strides=1, padding='same', activation='tanh', name=f'{self.name}_deconv4')
        ]

    def call(self, inputs, modulation_index, training=True):
        x = inputs
        for layer in self.initial_layers:
            if isinstance(layer, SwitchableBatchNormalization):
                x = layer(x, modulation_index, training=training)
            else:
                x = layer(x)
        for res_block in self.res_blocks:
            x = res_block(x, modulation_index, training=training)
        for layer in self.upsample_layers:
            if isinstance(layer, SwitchableBatchNormalization):
                x = layer(x, modulation_index, training=training)
            else:
                x = layer(x)
        for layer in self.final_layers:
            if isinstance(layer, SwitchableBatchNormalization):
                x = layer(x, modulation_index, training=training)
            else:
                x = layer(x)
        return x

class InnerEncoder(Model):
    def __init__(self, input_dim, output_dim, num_modulations, name='inner_encoder', **kwargs):
        super(InnerEncoder, self).__init__(name=name, **kwargs)
        self.conv = layers.Conv2D(filters=output_dim, kernel_size=5, strides=1, padding='same', activation=None, name=f'{self.name}_conv')
        self.sbn = SwitchableBatchNormalization(output_dim, num_modulations, name=f'{self.name}_sbn')
        self.tanh = layers.Activation('tanh')

    def call(self, inputs, modulation_index, training=True):
        x = self.conv(inputs)
        x = self.sbn(x, modulation_index, training=training)
        x = self.tanh(x)
        x = tf.reshape(x, [tf.shape(x)[0], -1, tf.shape(x)[-1]])
        return x

class InnerDecoder(Model):
    def __init__(self, input_dim, output_dim, num_modulations, name='inner_decoder', **kwargs):
        super(InnerDecoder, self).__init__(name=name, **kwargs)
        self.deconv = layers.Conv2DTranspose(filters=output_dim, kernel_size=5, strides=1, padding='same', activation=None, name=f'{self.name}_deconv')
        self.sbn = SwitchableBatchNormalization(output_dim, num_modulations, name=f'{self.name}_sbn')
        self.relu = layers.ReLU()

    def call(self, inputs, modulation_index, output_shape, training=True):
        x = tf.reshape(inputs, output_shape)
        x = self.deconv(x)
        x = self.sbn(x, modulation_index, training=training)
        x = self.relu(x)
        return x


class VectorQuantizer(Model):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5, **kwargs):
        super(VectorQuantizer, self).__init__(**kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        initializer = tf.random_uniform_initializer()
        self.embeddings = self.add_weight(
            name="embeddings",
            shape=(self.num_embeddings, self.embedding_dim),
            initializer=initializer,
            trainable=False,
        )
        self.ema_cluster_size = self.add_weight(
            name="ema_cluster_size",
            shape=(self.num_embeddings,),
            initializer=tf.zeros_initializer(),
            trainable=False,
        )
        self.ema_dw = self.add_weight(
            name="ema_dw",
            shape=(self.num_embeddings, self.embedding_dim),
            initializer=tf.zeros_initializer(),
            trainable=False,
        )

    def call(self, inputs, training=True):
        input_shape = tf.shape(inputs)
        flat_inputs = tf.reshape(inputs, [-1, self.embedding_dim])

        encoding_indices = self.get_code_indices(flat_inputs)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.nn.embedding_lookup(self.embeddings, encoding_indices)
        quantized = tf.reshape(quantized, input_shape)

        quantized = inputs + tf.stop_gradient(quantized - inputs)
        return quantized, encoding_indices, flat_inputs

    def update_codebook(self, encodings, flat_inputs):
        updated_ema_cluster_size = self.ema_cluster_size * self.decay + \
            (1 - self.decay) * tf.reduce_sum(encodings, axis=0)
        self.ema_cluster_size.assign(updated_ema_cluster_size)

        dw = tf.matmul(encodings, flat_inputs, transpose_a=True)
        updated_ema_dw = self.ema_dw * self.decay + (1 - self.decay) * dw
        self.ema_dw.assign(updated_ema_dw)

        n = tf.reduce_sum(self.ema_cluster_size)
        cluster_size = ((self.ema_cluster_size + self.epsilon) /
                        (n + self.num_embeddings * self.epsilon)) * n
        embed_normalized = self.ema_dw / tf.reshape(cluster_size, [-1, 1])
        self.embeddings.assign(embed_normalized)

    def get_code_indices(self, flat_inputs):
        distances = (tf.reduce_sum(flat_inputs ** 2, axis=1, keepdims=True)
                     - 2 * tf.matmul(flat_inputs, self.embeddings, transpose_b=True)
                     + tf.reduce_sum(self.embeddings ** 2, axis=1))
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices

    def embed_code(self, encoding_indices, enc_shape):
        quantized = tf.nn.embedding_lookup(self.embeddings, encoding_indices)
        return tf.reshape(quantized, enc_shape)
