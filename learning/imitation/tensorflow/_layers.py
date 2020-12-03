import tensorflow as tf

L2_LAMBDA = 1e-04


def _residual_block(x, size, dropout=False, dropout_prob=0.5, seed=None):
    residual = tf.layers.batch_normalization(x)  # TODO: check if the defaults in Tf are the same as in Keras
    residual = tf.nn.relu(residual)
    residual = tf.layers.conv2d(
        residual,
        filters=size,
        kernel_size=3,
        strides=2,
        padding="same",
        kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
        kernel_regularizer=tf.keras.regularizers.l2(L2_LAMBDA),
    )
    if dropout:
        residual = tf.nn.dropout(residual, dropout_prob, seed=seed)
    residual = tf.layers.batch_normalization(residual)
    residual = tf.nn.relu(residual)
    residual = tf.layers.conv2d(
        residual,
        filters=size,
        kernel_size=3,
        padding="same",
        kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
        kernel_regularizer=tf.keras.regularizers.l2(L2_LAMBDA),
    )
    if dropout:
        residual = tf.nn.dropout(residual, dropout_prob, seed=seed)

    return residual


def one_residual(x, keep_prob=0.5, seed=None):
    nn = tf.layers.conv2d(
        x,
        filters=32,
        kernel_size=5,
        strides=2,
        padding="same",
        kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
        kernel_regularizer=tf.keras.regularizers.l2(L2_LAMBDA),
    )
    nn = tf.layers.max_pooling2d(nn, pool_size=3, strides=2)

    rb_1 = _residual_block(nn, 32, dropout_prob=keep_prob, seed=seed)

    nn = tf.layers.conv2d(
        nn,
        filters=32,
        kernel_size=1,
        strides=2,
        padding="same",
        kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
        kernel_regularizer=tf.keras.regularizers.l2(L2_LAMBDA),
    )
    nn = tf.keras.layers.add([rb_1, nn])

    nn = tf.layers.flatten(nn)

    return nn
