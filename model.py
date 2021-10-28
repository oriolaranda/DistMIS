import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def down_block(input_tensor, num_filters, norm='bn', k=3):
    conv1 = conv_block(input_tensor, num_filters=num_filters, norm=norm, k=k)
    conv2 = conv_block(conv1, num_filters=num_filters, norm=norm, k=k)

    max_pool = tf.keras.layers.MaxPooling3D(
        pool_size=(2, 2, 2), strides=2,
        data_format='channels_first')(conv2)
    return max_pool, conv2


def up_block(input_tensor, shortcut, num_filters, norm='bn', k=3):
    deconv = tf.keras.layers.Conv3DTranspose(
        filters=num_filters, kernel_size=(k,) * 3, strides=(2, 2, 2),
        padding='same', data_format='channels_first')(input_tensor)

    concat = tf.concat(values=[deconv, shortcut], axis=1, name="concat")
    conv1 = conv_block(concat, num_filters=num_filters, norm=norm, k=k)
    conv2 = conv_block(conv1, num_filters=num_filters, norm=norm, k=k)
    return conv2


def conv_block(input_tensor, num_filters, norm='bn', k=3):
    """
    Convolution and batch normalization layer
    :param input_tensor: The input tensor
    :param is_training: Boolean tensor whether it is being run on training or not
    :param num_filters: The number of filters to convolve on the input
    :param name: Name of the convolutional block
    :return: Tensor after convolution and batch normalization
    """
    kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=5e-2)
    bias_initializer = tf.zeros_initializer()

    conv = tf.keras.layers.Conv3D(filters=num_filters, kernel_size=(k,) * 3, strides=(1, 1, 1), padding='same',
                                  data_format='channels_first', activation=None, use_bias=True,
                                  kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(
        input_tensor)
    if norm == 'bn':
        # Batch normalization before activation
        nl = tf.keras.layers.BatchNormalization(
            axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True)(conv)
    elif norm == 'gn':
        nl = tfa.layers.GroupNormalization(groups=19)(conv)
    elif norm == 'none':
        nl = conv
    else:
        print('ERROR in Normalization!!!')

    # Activation after normalization
    act = tf.nn.relu(nl)
    return act


def model_unet_tune(input_shape, filter_start=8, loss_function=None, metrics=None,
                    initial_learning_rate=0.0001, amsgrad=False, b2=0.9, depth=3, norm='bn', k=3):
    input_layer = Input(input_shape)

    level1, l1_conv = down_block(input_layer, num_filters=filter_start, norm=norm, k=k)
    level2, l2_conv = down_block(level1, num_filters=filter_start * 2, norm=norm, k=k)
    if depth == 3:
        level3, l3_conv = down_block(level2, num_filters=filter_start * 4, norm=norm, k=k)
        conv1 = conv_block(level3, num_filters=filter_start * 8, norm=norm, k=k)
        conv2 = conv_block(conv1, num_filters=filter_start * 8, norm=norm, k=k)
        level3_up = up_block(conv2, l3_conv, num_filters=filter_start * 4, norm=norm, k=k)
    else:
        conv1 = conv_block(level2, num_filters=filter_start * 4, norm=norm, k=k)
        level3_up = conv_block(conv1, num_filters=filter_start * 4, norm=norm, k=k)

    level2_up = up_block(level3_up, l2_conv, num_filters=filter_start * 2, norm=norm, k=k)
    level1_up = up_block(level2_up, l1_conv, num_filters=filter_start, norm=norm, k=k)

    kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=5e-2)
    bias_initializer = tf.zeros_initializer()

    output = tf.keras.layers.Conv3D(filters=1, kernel_size=(k,) * 3, strides=(1, 1, 1), padding='same',
                                    data_format='channels_first', activation=tf.nn.sigmoid, use_bias=True,
                                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(level1_up)

    model = Model(inputs=input_layer, outputs=output, name='3DUnet')

    if not isinstance(metrics, list):
        metrics = [metrics]

    model.compile(optimizer=Adam(lr=initial_learning_rate, beta_2=b2, amsgrad=amsgrad), loss=loss_function,
                  metrics=metrics)
    return model
