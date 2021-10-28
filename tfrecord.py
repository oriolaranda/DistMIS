########################################################################################################################
# @author Oriol Aranda (https://github.com/oriolaranda/)
# @date Oct 2021
########################################################################################################################

import argparse
import json
from functools import partial
from os import path
import numpy as np
import nibabel as nib
import tensorflow as tf
from tqdm import tqdm
from itertools import accumulate
from operator import add
from utils import resize_image


@tf.autograph.experimental.do_not_convert
def brain_dataset(sample, source_dir, verbose=0):
    def _generator(names):
        image_name, label_name = names
        if verbose:
            print("Training on sample:", source_dir + str(image_name[2:], 'utf-8'))
        image_dir = source_dir + str(image_name[2:], 'utf-8')
        label_dir = source_dir + str(label_name[2:], 'utf-8')
        x = np.array(nib.load(image_dir).get_fdata())[:, :, 2:-1, :]
        y = np.array(nib.load(label_dir).get_fdata())[:, 2:-1, :]
        y_ = np.zeros(y.shape)
        y_[(y > 0) & (y < 4)] = 1
        x = np.moveaxis(x, -1, 0)
        y = np.expand_dims(y_, -1)
        y = np.moveaxis(y, -1, 0)
        yield x, y

    dataset = tf.data.Dataset.from_generator(
        _generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=((4, 240, 240, 152), (1, 240, 240, 152)),
        args=(sample,))
    return dataset


def sets_creator(data, datasets_p, source_dir, resize_shape):
    def dataset_gen(samples):
        def preproc_fn(x, y):
            if resize_shape != (240, 240, 152):
                assert len(resize_shape) == 3 and all(s > 0 for s in resize_shape), \
                    f"Resize shape is wrong! {resize_shape}?"
                x, y = resize_image(x, y, resize_shape)
            x = tf.image.per_image_standardization(x)
            return x, y

        brain_mri_dataset = partial(brain_dataset, source_dir=source_dir)
        _dataset = tf.data.Dataset.from_tensor_slices(samples)
        _dataset = _dataset.interleave(lambda x: brain_mri_dataset(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        _dataset = _dataset.map(preproc_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return _dataset

    # nth position to split each set; accumulate probabilities to calculate each n
    train_n, valid_n, test_n = (int(p * len(data)) for p in accumulate(datasets_p, add))
    split_samples = data[:train_n], data[train_n:valid_n], data[valid_n:test_n]
    train, valid, test = ((dataset_gen(samples), len(samples)) for samples in split_samples)
    return train, valid, test


def _bytes_feature(value):
    """Returns a bytes_list from a string / bytes."""
    if isinstance(value, type(tf.constant(0))):  # if value ist tensor
        value = value.numpy()  # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_sample(img, label):
    """ Creates a tf.train.Example message ready to be written to a file."""
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible data type.
    features = {
        'img': _bytes_feature(tf.io.serialize_tensor(img)),
        'label': _bytes_feature(tf.io.serialize_tensor(label)),
    }
    # Create a Features message using tf.train.Example
    sample = tf.train.Example(features=tf.train.Features(feature=features))
    return sample.SerializeToString()


def serialize_dataset(dataset_generator, dataset_size):
    def serialize_gen():
        for sample in tqdm(dataset_generator, total=dataset_size):
            yield serialize_sample(*sample)

    return serialize_gen


def _write_dataset(dataset, name, dataset_size, num_shards, target_dir):
    for i in range(num_shards):
        shard_dataset = dataset.shard(num_shards=num_shards, index=i)
        serialized_shard = tf.data.Dataset.from_generator(serialize_dataset(shard_dataset, dataset_size // num_shards),
                                                          output_types=tf.string, output_shapes=())
        writer = tf.data.experimental.TFRecordWriter(target_dir + f"{name}_{i}.tfrecord")
        writer.write(serialized_shard)
        print(f"TFRecord {name}_{i} saved!")
    print(f"TFRecords for {name} written!!")


def _write_info(info, target_dir):
    json_path = path.join(target_dir, 'info.json')
    with open(json_path, 'w') as f:
        json.dump(info, f)
    print("Datasets info written!")


def set_dir(*funcs, target):
    return tuple(partial(f, target_dir=target) for f in funcs)


def main(args):
    source_json = path.join(args.source_dir, "dataset.json")

    assert path.exists(args.source_sir), f"The source dir couldn't be found! {args.source_dir}"
    assert path.exists(source_json), f"Json file in the source dir couldn't be found! {source_json}"
    assert len(args.split) == 3 and sum(args.split) == 1, f"Split arguments does not sum up to 1: {args.split}"

    with open(source_json) as f:
        dataset = json.load(f)
    data = [(d['image'], d['label']) for d in dataset['training']]

    (train, valid, test), sizes = zip(*sets_creator(data, tuple(args.split), args.source_dir, tuple(args.reshape)))

    sizes = dict(zip(('train_size', 'valid_size', 'test_size'), sizes))
    shards = dict(zip(('train_shard', 'valid_shard', 'test_shard'), (16, 4, 4)))
    info = {"total_size": len(data), **sizes, **shards}

    if args.target_dir:
        assert path.exists(args.target_dir), "Target dir doesn't exist!"
        write_dataset, write_info = set_dir(_write_dataset, _write_info, target=args.target_dir)
        write_dataset(train, 'train', info['train_size'], info['train_shard'])
        write_dataset(valid, 'valid', info['valid_size'], info['valid_shard'])
        write_dataset(test, 'test', info['test_size'], info['test_shard'])
        write_info(info)
        print(f"Done!! The entire dataset has been written in TFRecord format in '{args.target_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=str, required=True,
                        help="Path: Source data directory. The directory must contain the dataset.json file,"
                             "and the two folders imagesTr, labelsTr.")
    parser.add_argument("--target-dir", type=str, default=None,
                        help="Path: Target data directory. It must exist. It is where the TFRecord data will be "
                             "written to")
    parser.add_argument("--split", type=tuple, default=(0.7, 0.15, 0.15),
                        help="Tuple: Ratios into which the dataset will be divided (all sum up to 1). "
                             "Train, validation and test set. Default=(0.7, 0.15, 0.15).")
    parser.add_argument("--reshape", type=tuple, default=(240, 240, 152),
                        help="Tuple: Shape of the written data. Default (240, 240, 152) is the original shape, so no "
                             "resize will be applied. ")
    _args, _ = parser.parse_known_args()

    main(_args)
