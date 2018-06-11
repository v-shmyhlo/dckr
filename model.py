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
        input = tf.layers.dropout(input, 0.2, noise_shape=(shape[0], 1, 1, shape[3]), training=training)

        return input

    def downsample(input):
        input = tf.layers.max_pooling2d(input, 2, 2)
        return input

    def upsample(input):
        input = tf.image.resize_nearest_neighbor(input, (input.shape[1] * 2, input.shape[2] * 2), align_corners=True)
        return input

    def dense_down_up(input, filters):
        batch_size = tf.shape(input)[0]
        shape = input.shape

        input = tf.reshape(input, (batch_size, shape[1] * shape[2] * shape[3]))
        input = tf.layers.dense(
            input, filters,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        input = tf.nn.elu(input)

        latent = input

        input = tf.layers.dense(
            input, shape[1] * shape[2] * shape[3],
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        input = tf.nn.elu(input)
        input = tf.reshape(input, (batch_size, shape[1], shape[2], shape[3]))

        return input, latent

    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)
    num_channels = input.shape[-1]

    input = conv_bn(input, 32)
    input = downsample(input)
    input = conv_bn(input, 64)
    input = downsample(input)
    input = conv_bn(input, 128)
    # input = downsample(input)
    # input = conv_bn(input, 128)

    input, latent = dense_down_up(input, 32)

    # input = conv_bn(input, 64)
    # input = upsample(input)
    input = conv_bn(input, 128)
    input = upsample(input)
    input = conv_bn(input, 64)
    input = upsample(input)
    input = conv_bn(input, 32)

    # output layer
    input = conv(input, num_channels)

    return input, latent
