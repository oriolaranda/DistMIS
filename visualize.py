########################################################################################################################
# @author Oriol Aranda (https://github.com/oriolaranda/Distributing-Deep-Learning-3D-Medical-Image-Segmentation)
# @date Oct 2021
########################################################################################################################
import argparse
import json
from functools import partial
from os import path
import numpy as np
import imageio
import tensorflow as tf
from matplotlib import animation, pyplot as plt
from utils import parse_raw_sample


def visualize_gif(data_, name='visual'):
    """
    Save a gif image for the labeled image. Modified for unet ouputs.
    """
    data_ = data_.squeeze()
    images = []
    for i in range(data_.shape[0]):
        x = data_[min(i, data_.shape[0] - 1), :, :]
        y = data_[:, min(i, data_.shape[1] - 1), :]
        z = data_[:, :, min(i, data_.shape[2] - 1)]
        img = np.concatenate((x, y, z), axis=1)
        images.append(img)
    imageio.mimsave(f"./gif_{name}.gif", images, duration=0.1)


def visualize_gif_graph(data_):
    data_ = data_.squeeze()
    fig = plt.figure()
    images = []
    for i in range(data_.shape[0]):
        x = data_[min(i, data_.shape[0] - 1), :, :]
        y = data_[:, min(i, data_.shape[1] - 1), :]
        z = data_[:, :, min(i, data_.shape[2] - 1)]
        img = np.concatenate((x, y, z), axis=1)
        im = plt.imshow(img, animated=True)
        images.append([im])
    ani = animation.ArtistAnimation(fig, images, interval=100, blit=True)
    plt.show()


def labeled_image(image, label):
    """
    Merge image and label and return a labeled_image for 1 label class.
    """
    image = image[0, :, :, :]
    image = (image - image.min())*255/(image.max()-image.min())
    # image = image.astype(np.uint8)
    # label = label.astype(np.uint8)
    image, label = image.astype(np.uint8), label.astype(np.uint8)
    labeled_img = np.zeros_like(label[:, :, :, :])
    # remove tumor part from image
    labeled_img[0, :, :, :] = image * (1 - label[0, :, :, :])
    # color labels
    labeled_img += label[0, :, :, :] * 255
    return labeled_img


def data_generator(file_path, data_shape):
    filenames = [file_path]
    raw_train_dataset = tf.data.TFRecordDataset(filenames)

    _parse_raw_sample = parse_raw_sample(data_shape)

    def gen_parse_raw_sample():
        for raw_sample in raw_train_dataset:
            yield _parse_raw_sample(raw_sample)

    _dataset = tf.data.Dataset.from_generator(
        gen_parse_raw_sample,
        output_types=(tf.float32, tf.float32),
        output_shapes=((4, *data_shape), (1, *data_shape))
    )
    _dataset = _dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return _dataset


def which_file(idx, info):
    assert 0 <= idx < info['total_size'], f"Sample number out of range. 0 <= i < {info['total_size']}"

    if idx < info['train_size']:
        _set, _idx = "train", idx
    elif idx < info['train_size'] + info['valid_size']:
        _set, _idx = "valid", idx - info['train_size']
    else:
        _set, _idx = "test", idx - info['train_size'] - info['valid_size']
    return _set, divmod(_idx, info[f'{_set}_size']//info[f'{_set}_shard'])


def main(args):
    assert path.exists(args.dataset_dir), f"The dataset dir is wrong: {args.dataset_dir}"
    info_json = path.join(args.dataset_dir, "info.json")
    assert path.exists(info_json), f"The dataset dir is wrong! '{info_json}' not found!"

    with open(info_json) as f:
        info = json.load(f)
    file_set, (file_num, pos) = which_file(args.sample, info)
    file = path.join(args.dataset_dir, f"{file_set}_{file_num}.tfrecord")

    rec_dataset = data_generator(file, args.data_shape)
    img, label = list(rec_dataset.as_numpy_iterator())[pos]
    if args.no_screen:
        visualize_gif(labeled_image(img.copy(), label.copy()), f'sample_{args.sample}')
        print('GIF created!')
    else:
        visualize_gif_graph(labeled_image(img.copy(), label.copy()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True,
                        help="Path: TFRecord dataset directory.")
    parser.add_argument("--sample", type=int, default=0,
                        help="Int: Sample to visualize. Default=0. It has to be 0<= sample <= size_dataset.")
    parser.add_argument("--data-shape", type=str, default=(240, 240, 152),
                        help="Tuple: Shape of the data in the dataset path provided. Default=(240, 240, 152)"
                             " which is the orginal data shape.")
    parser.add_argument("--no-screen", default=False, action='store_true',
                        help="Bool: No X session (graphical Linux desktop) mode. Default=False. If set to "
                             "True, a GIF file will be saved in './' containing the plotted image.")
    _args, _ = parser.parse_known_args()

    main(_args)
