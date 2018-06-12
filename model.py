import tensorflow as tf


def model(input, training):
    def conv(input, filters):
        input = tf.layers.conv2d(
            input, filters, 3,
            padding='same',
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        return input

    def conv_bn(input, filters):
        input = conv(input, filters)
        input = tf.layers.batch_normalization(input, training=training)
        input = tf.nn.elu(input)
        shape = tf.shape(input)
        input = tf.layers.dropout(input, dropout_rate, noise_shape=(shape[0], 1, 1, shape[3]), training=training)
        return input

    def fc_bn(input, filters):
        input = tf.layers.dense(
            input, filters,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        input = tf.layers.batch_normalization(input, training=training)
        input = tf.nn.elu(input)
        input = tf.layers.dropout(input, dropout_rate, training=training)
        return input

    def downsample(input):
        input = tf.layers.max_pooling2d(input, 2, 2)
        return input

    def upsample(input):
        input = tf.image.resize_nearest_neighbor(input, (input.shape[1] * 2, input.shape[2] * 2), align_corners=True)
        return input

    def reshape_to_vector(input):
        b = tf.shape(input)[0]
        _, h, w, c = input.shape
        input = tf.reshape(input, (b, h * w * c))
        return input, (b, h, w, c)

    def reshape_to_image(input, shape):
        input = tf.reshape(input, shape)
        return input

    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)
    num_channels = input.shape[-1]
    dropout_rate = 0.2

    # encoder
    input = conv_bn(input, 32)
    input = downsample(input)
    input = conv_bn(input, 64)
    input, shape = reshape_to_vector(input)
    input = latent = fc_bn(input, 32)

    # decoder
    input = fc_bn(input, shape[1] * shape[2] * shape[3])
    input = reshape_to_image(input, shape)
    input = conv_bn(input, 32)
    input = upsample(input)
    input = conv(input, num_channels)

    return input, latent
