########################################################################################################################
# @author Oriol Aranda (https://github.com/oriolaranda/)
# @date Oct 2021
########################################################################################################################

import tensorflow as tf
from ray import tune


###############################
# Metrics and Loss functions
###############################

def dice_coeff(seg_true, seg_pred, smooth=0.01):
    seg_true_flat = tf.keras.layers.Flatten(data_format='channels_first', dtype=tf.float32)(seg_true)
    seg_pred_flat = tf.keras.layers.Flatten(data_format='channels_first', dtype=tf.float32)(seg_pred)

    intersection = tf.multiply(seg_true_flat, seg_pred_flat, name="intersection")
    intersect = tf.reduce_sum(intersection)
    dice = tf.divide(2 * intersect + smooth, tf.reduce_sum(seg_true_flat) + tf.reduce_sum(seg_pred_flat) + smooth)
    return dice


def dice_loss(seg_true, seg_pred):
    return 1 - dice_coeff(seg_true, seg_pred)


def loss_metric(loss, metric):
    @tf.autograph.experimental.do_not_convert
    def loss_func(seg_true, seg_pred):
        return loss(seg_true, seg_pred)

    @tf.autograph.experimental.do_not_convert
    def metric_func(seg_true, seg_pred):
        return metric(seg_true, seg_pred)

    return loss_func, metric_func


##############
# Data utils
##############

@tf.autograph.experimental.do_not_convert
def parse_raw_sample(data_shape):
    def _parse_raw_sample(sample):
        parse_dic = {
            'img': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.string)
        }
        sample_message = tf.io.parse_single_example(sample, parse_dic)

        img = sample_message['img']  # get byte string
        label = sample_message['label']
        img = tf.io.parse_tensor(img, out_type=tf.float32)  # restore 2D array from byte string
        label = tf.io.parse_tensor(label, out_type=tf.float32)
        img = tf.ensure_shape(img, (4, *tuple(data_shape)))
        label = tf.ensure_shape(label, (1, *tuple(data_shape)))
        return img, label

    return _parse_raw_sample


def data_generator(filenames, config, shuffle_size, train=False):
    raw_dataset = tf.data.TFRecordDataset(filenames,  # buffer_size=config['tfrecord_buffer_size'],
                                          num_parallel_reads=config['tfrecord_num_parallel_reads'])
    parsed_dataset = raw_dataset.map(parse_raw_sample(config['input_shape']),
                                     num_parallel_calls=config['map_num_parallel_calls'])

    if train:
        parsed_dataset = parsed_dataset.shuffle(shuffle_size).repeat().batch(config['batch_size'])
    else:
        parsed_dataset = parsed_dataset.repeat(1).batch(config['batch_size'])
    parsed_dataset = parsed_dataset.prefetch(config['prefetch_buffer_size'])
    return parsed_dataset


##############
# Callbacks
##############

class TuneReporter(tf.keras.callbacks.Callback):
    """Tune Callback for Keras."""

    def __init__(self, reporter=None, freq="epoch", logs=None):
        super(TuneReporter, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        tune.report(keras_info=logs, loss=logs['loss'], dice_coefficient=logs['metric_func'])

    def on_test_end(self, logs=None):
        tune.report(keras_info=logs, val_loss=logs['loss'],
                    val_dice_coefficient=logs['metric_func'])


################################################
# Data Augmentation and Preprocessing functions
################################################


def _resize_by_axis_trilinear(images, size_0, size_1, ax):
    """
    Resize image bilinearly to [size_0, size_1] except axis ax.
        :param image: a tensor 4-D with shape
                        [batch, d0, d1, d2, channels]
        :param size_0: size 0
        :param size_1: size 1
        :param ax: axis to exclude from the interpolation
    """
    resized_list = []
    # unstack the image in 2d cases
    unstack_list = tf.unstack(images, axis=ax)
    for i in unstack_list:
        resized_list.append(tf.image.resize(i, [size_0, size_1]))  # resize bilinearly
    stack_img = tf.stack(resized_list, axis=ax)
    return stack_img


def resize_trilinear(images, size):
    """
    Resize images to size using trilinear interpolation.
        :param images: A tensor 5-D with shape
                        [batch, d0, d1, d2, channels]
        :param size: A 1-D int32 Tensor of 3 elements: new_d0, new_d1,
                        new_d2. The new size for the images.
    """
    assert size.shape[0] == 3
    x_ = tf.expand_dims(images, axis=0)
    x_ = tf.transpose(x_, [0, 2, 3, 4, 1])
    resized = _resize_by_axis_trilinear(x_, size[0], size[1], 3)
    resized = _resize_by_axis_trilinear(resized, size[0], size[2], 1)
    resized = tf.transpose(resized, [0, 4, 1, 2, 3])
    resized = tf.squeeze(resized, axis=[0])
    return resized


def resize_image(x, y, resize_shape):
    x_ = resize_trilinear(x, tf.constant(list(resize_shape)))
    y_ = resize_trilinear(y, tf.constant(list(resize_shape)))
    return x_, y_


def data_augmentation(flip=False, noise=False, rotation=False, p=0.1):
    """Apply data augmentation: add noise, flip nad rotation"""
    print("Data Augmentation:", flip, noise, p)

    def _apply_transforms(*sample):
        probs = tf.random.uniform([5])
        if noise and probs[0] < p:
            sample = _add_noise(*sample)
        if flip:
            if probs[1] < p:
                sample = _flip_up_down(*sample)
            if probs[2] < p:
                sample = _flip_left_right(*sample)  # the most sense flip
            if probs[3] < p:
                sample = _flip_front_back(*sample)
        if rotation and probs[4] < p:
            pass
        img, label = sample
        return img, label

    return _apply_transforms


@tf.function
def _flip(img, label, axis):
    flipped_img = tf.reverse(img, axis=[axis])
    flipped_label = tf.reverse(label, axis=[axis])
    return flipped_img, flipped_label


def _flip_up_down(img, seg):
    return _flip(img, seg, 3)


def _flip_left_right(img, label):
    return _flip(img, label, 1)


def _flip_front_back(img, label):
    return _flip(img, label, 2)


@tf.function
def _add_noise(img, label):
    mean, std = tf.nn.moments(img, axes=[0, 1, 2, 3])
    noise = tf.random.normal(shape=img.shape, mean=0.0, stddev=0.2, dtype=tf.float32)
    # non_zero = img > 0
    # _img = tf.where(non_zero, img + noise, img)
    _img = img + noise
    st_img = tf.image.per_image_standardization(_img)
    return st_img, label





