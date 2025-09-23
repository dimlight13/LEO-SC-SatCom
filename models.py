import tensorflow as tf
from keras import layers, Model, losses
from keras import applications

class VGGFeatureMatchingLoss(losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder_layers = [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ]
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        vgg = applications.VGG19(include_top=False, weights="imagenet")
        layer_outputs = [vgg.get_layer(x).output for x in self.encoder_layers]
        self.vgg_model = Model(vgg.input, layer_outputs, name="VGG")
        self.mae = losses.MeanAbsoluteError()

    def call(self, y_true, y_pred):
        y_true = applications.vgg19.preprocess_input(127.5 * (y_true + 1))
        y_pred = applications.vgg19.preprocess_input(127.5 * (y_pred + 1))
        real_features = self.vgg_model(y_true)
        fake_features = self.vgg_model(y_pred)
        loss = 0
        for i in range(len(real_features)):
            loss += self.weights[i] * self.mae(real_features[i], fake_features[i])
        return loss

class VQVAE(Model):
    def __init__(self, num_embeddings, embedding_dim, num_modulations, 
                 commitment_cost=0.25, decay=0.99, n_res_block=2, img_size=32, name='vqvae', **kwargs):
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

        self.encoder = Encoder(channels, n_res_block, num_modulations, img_size)
        self.inner_encoder = InnerEncoder(embedding_dim, num_modulations)
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost, decay)
        self.inner_decoder = InnerDecoder(channels[-1], num_modulations)
        self.decoder = Decoder(channels[::-1], n_res_block, num_modulations, img_size)

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
        symbols = tf.reshape(symbols, (-1, ))
        bits = demodulate_fn(symbols, M_mod)
        return bits

    def encode(self, x, modulation_index, training=False):
        modulation_index = tf.cast(modulation_index, tf.int32)
        z = self.encoder(x, modulation_index, training=training)
        encoded_features = self.inner_encoder(z, modulation_index, training=training)
        return encoded_features, tf.shape(z)

    def quantize(self, z, training=False):
        quantized, encoding_indices, flat_inputs = self.vq_layer(z, training=training)
        return quantized, encoding_indices, flat_inputs

    def decode(self, recovered_bits, modulation_index, z_shape, training=False):
        decoding_vector = tf.reshape(recovered_bits, (z_shape[0], z_shape[1], z_shape[2], self.embedding_dim))
        decoding_vector = self.inner_decoder(decoding_vector, modulation_index, training=training)

        x_recon = self.decoder(decoding_vector, modulation_index, training=training)
        return x_recon

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

        gamma = tf.reshape(gamma, [1, 1, 1, -1])
        beta = tf.reshape(beta, [1, 1, 1, -1])

        if training:
            batch_mean, batch_variance = tf.nn.moments(inputs, axes=[0, 1, 2], keepdims=False)
            self.moving_mean[modulation_index].assign(
                self.momentum * moving_mean + (1 - self.momentum) * batch_mean)
            self.moving_variance[modulation_index].assign(
                self.momentum * moving_variance + (1 - self.momentum) * batch_variance)
            mean = batch_mean
            variance = batch_variance
        else:
            mean = moving_mean
            variance = moving_variance

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
    def __init__(self, channels, n_res_block, num_modulations, img_size, name='encoder', **kwargs):
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
        
        if img_size == 32:
            self.final_layers = [
                layers.Conv2D(filters=channels[2], kernel_size=2, strides=2, padding='valid', activation=None, name=f'{self.name}_conv3'),
                SwitchableBatchNormalization(channels[2], num_modulations, name=f'{self.name}_sbn3'),
                layers.ReLU()
            ]
        elif img_size == 64:
            self.final_layers = [
                layers.Conv2D(filters=channels[2], kernel_size=2, strides=2, padding='valid', activation=None, name=f'{self.name}_conv3'),
                SwitchableBatchNormalization(channels[2], num_modulations, name=f'{self.name}_sbn3'),
                layers.ReLU(),
                layers.Conv2D(filters=channels[2], kernel_size=2, strides=2, padding='valid', activation=None, name=f'{self.name}_conv4'),
                SwitchableBatchNormalization(channels[2], num_modulations, name=f'{self.name}_sbn4'),
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

class InnerEncoder(Model):
    def __init__(self, output_dim, num_modulations, name='inner_encoder', **kwargs):
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

class Decoder(Model):
    def __init__(self, channels, n_res_block, num_modulations, img_size, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.num_modulations = num_modulations

        self.initial_layers = [
            layers.Conv2DTranspose(filters=channels[0], kernel_size=3, strides=1, padding='same', activation=None, name=f'{self.name}_deconv1'),
            SwitchableBatchNormalization(channels[0], num_modulations, name=f'{self.name}_sbn1'),
            layers.ReLU()
        ]

        self.res_blocks = [ResBlock(channels[0], num_modulations, name=f'{self.name}_resblock_{i}') for i in range(n_res_block)]
        
        if img_size == 32:
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

        elif img_size == 64:
            self.upsample_layers = [
                layers.Conv2DTranspose(filters=channels[1], kernel_size=2, strides=2, padding='valid', activation=None, name=f'{self.name}_deconv2'),
                SwitchableBatchNormalization(channels[1], num_modulations, name=f'{self.name}_sbn2'),
                layers.ReLU(),
                layers.Conv2DTranspose(filters=channels[1], kernel_size=2, strides=2, padding='valid', activation=None, name=f'{self.name}_deconv3'),
                SwitchableBatchNormalization(channels[1], num_modulations, name=f'{self.name}_sbn3'),
                layers.ReLU()
            ]
            self.final_layers = [
                layers.Conv2DTranspose(filters=channels[2], kernel_size=5, strides=1, padding='same', activation=None, name=f'{self.name}_deconv4'),
                SwitchableBatchNormalization(channels[2], num_modulations, name=f'{self.name}_sbn4'),
                layers.ReLU(),
                layers.Conv2DTranspose(filters=3, kernel_size=5, strides=1, padding='same', activation='tanh', name=f'{self.name}_deconv5')
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

class InnerDecoder(Model):
    def __init__(self, output_dim, num_modulations, name='inner_decoder', **kwargs):
        super(InnerDecoder, self).__init__(name=name, **kwargs)
        self.deconv = layers.Conv2DTranspose(filters=output_dim, kernel_size=5, strides=1, padding='same', activation=None, name=f'{self.name}_deconv')
        self.sbn = SwitchableBatchNormalization(output_dim, num_modulations, name=f'{self.name}_sbn')
        self.relu = layers.ReLU()

    def call(self, inputs, modulation_index, training=True):
        x = self.deconv(inputs)
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


class BasicBlock(tf.keras.Model):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(planes, kernel_size=3, strides=stride, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(planes, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()

        self.shortcut = tf.keras.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut.add(layers.Conv2D(self.expansion * planes, kernel_size=1, strides=stride, use_bias=False))
            self.shortcut.add(layers.BatchNormalization())

    def call(self, x, training=True):
        out = layers.ReLU()(self.bn1(self.conv1(x), training=training))
        out = self.bn2(self.conv2(out), training=training)
        out += self.shortcut(x, training=training)
        out = layers.ReLU()(out)
        return out

class Actor(tf.keras.Model):
    def __init__(self, action_dim=5):
        super().__init__(name="actor")
        self.conv1 = layers.Conv2D(64, 3, padding="same", use_bias=False)
        self.bn1   = layers.BatchNormalization()
        self.relu  = layers.ReLU()
        self.block1 = BasicBlock(64, 128, stride=2)
        self.block2 = BasicBlock(128, 256, stride=2)
        self.gap    = layers.GlobalAveragePooling2D()
        self.d1 = layers.Dense(64, activation="relu")
        self.d2 = layers.Dense(64, activation="relu")
        self.merge_dense = layers.Dense(128, activation="relu")
        self.logits = layers.Dense(action_dim, activation="linear")

    def call(self, inputs, training=False):
        img, snr, tau, fd = inputs
        x = self.conv1(img)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.gap(x)                 # [B,256]
        ch = tf.concat([snr, tau, fd], axis=-1)  # [B,1+5+5=11]
        ch = self.d1(ch)
        ch = self.d2(ch)                # [B,64]
        merged = tf.concat([x, ch], axis=-1)  # [B,320]
        merged = self.merge_dense(merged)     # [B,128]
        return self.logits(merged)            # [B, action_dim]


class Critic(tf.keras.Model):
    def __init__(self):
        super().__init__(name="critic")
        self.conv1 = layers.Conv2D(64, 3, padding="same", use_bias=False)
        self.bn1   = layers.BatchNormalization()
        self.relu  = layers.ReLU()
        self.block = BasicBlock(64, 128, stride=2)
        self.gap   = layers.GlobalAveragePooling2D()
        self.d1 = layers.Dense(64, activation="relu")
        self.d2 = layers.Dense(64, activation="relu")
        self.merge_dense = layers.Dense(64, activation="relu")
        self.value = layers.Dense(1, activation="linear")

    def call(self, inputs, training=False):
        img, snr, tau, fd = inputs
        x = self.conv1(img)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.block(x, training=training)
        x = self.gap(x)                 # [B,128]
        ch = tf.concat([snr, tau, fd], axis=-1)  # [B,11]
        ch = self.d1(ch)
        ch = self.d2(ch)                # [B,64]
        merged = tf.concat([x, ch], axis=-1)  # [B,192]
        merged = self.merge_dense(merged)     # [B,64]
        return self.value(merged)             # [B,1]

class ResidualBlock(Model):
    def __init__(self, hidden_dim, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.dense1 = layers.Dense(hidden_dim, activation='relu')
        self.dropout = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(hidden_dim)
        self.layer_norm = layers.LayerNormalization()

    def call(self, x, training=False):
        shortcut = x
        out = self.dense1(x)
        out = self.dropout(out, training=training)
        out = self.dense2(out)
        out += shortcut
        return self.layer_norm(out)

class PostLMMSENet(Model):
    def __init__(self, hidden_dim=256, mod_embed_dim=32, num_res_blocks=4, dropout_rate=0.05):
        super(PostLMMSENet, self).__init__()
        self.mod_embedding = layers.Embedding(input_dim=5, output_dim=mod_embed_dim)
        self.mod_proj = layers.Dense(hidden_dim, activation='relu')
        
        self.input_dense = layers.Dense(hidden_dim, activation='relu')
        
        self.res_blocks = [ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_res_blocks)]
        
        self.out_layer = layers.Dense(2, activation='linear')
        self.global_skip = layers.Dense(hidden_dim, activation=None)
        self.out_norm = layers.LayerNormalization()

    def call(self, eq_data, mod_idx, training=False):
        batch_size = tf.shape(eq_data)[0]
        num_symbols = tf.shape(eq_data)[1]
        
        real_part = tf.math.real(eq_data)
        imag_part = tf.math.imag(eq_data)
        x = tf.stack([real_part, imag_part], axis=-1)
        
        mod_idx = tf.reshape(mod_idx, [-1])  # (batch_size,)
        mod_emb = self.mod_embedding(mod_idx)  # (batch_size, mod_embed_dim)
        mod_emb = self.mod_proj(mod_emb)         # (batch_size, hidden_dim)
        mod_emb_2d = tf.reshape(mod_emb, [batch_size, 1, -1])  # (batch_size, 1, hidden_dim)
        mod_emb_2d = tf.tile(mod_emb_2d, [1, num_symbols, 1])   # (batch_size, num_symbols, hidden_dim)
        
        x = tf.reshape(x, [batch_size * num_symbols, 2])
        x = self.input_dense(x)  # (batch_size*num_symbols, hidden_dim)
        x = tf.reshape(x, [batch_size, num_symbols, -1])  # (batch_size, num_symbols, hidden_dim)
        
        x = x + mod_emb_2d
        
        global_feature = self.global_skip(x)
        
        for block in self.res_blocks:
            x = block(x, training=training)
        
        x = x + global_feature
        x = self.out_norm(x)
        
        out = self.out_layer(x)  # (batch_size, num_symbols, 2)
        
        eq_data_real = out[..., 0]
        eq_data_imag = out[..., 1]
        eq_data_post = tf.complex(eq_data_real, eq_data_imag)
        return eq_data_post
