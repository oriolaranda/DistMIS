########################################################################################################################
# @author Oriol Aranda (https://github.com/oriolaranda/)
# @date Oct 2021
########################################################################################################################
from os import environ, path
import time
import argparse
from datetime import datetime, timedelta
import json
import tensorflow as tf
from auxiliary_brain_seg import *
import ray
from itertools import product
from ray.util.sgd.tf.tf_trainer import TFTrainer, TFTrainable
from utils import loss_metric, data_generator


def data_creator(config):
    files_tr = [config['dataset_path'] + f"train_{i}.tfrecord" for i in range(config['train_shard'])]
    files_val = [config['dataset_path'] + f"valid_{i}.tfrecord" for i in range(config['valid_shard'])]

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

    train_g = data_generator(files_tr, config, config['train_size'] // 10, train=True)
    valid_g = data_generator(files_val, config, config['valid_size'] // 10)
    return train_g.with_options(options), valid_g.with_options(options)


def model_creator(config):
    loss, metric = loss_metric(*config['loss_metric'])
    model = model_unet_tune(input_shape=(4, 240, 240, 152), filter_start=8, loss_function=loss, metrics=[metric],
                            initial_learning_rate=config['lr'], amsgrad=config['amsgrad'], b2=config['b2'],
                            norm=config['norm'])
    return model


def create_config(config):
    train_size, valid_size = config['train_size'], config['valid_size']

    tstamp = "{}".format(datetime.now().strftime("%m_%d_%Y-%H:%M:%S"))
    logdir = f"{config['log_dir']}/logs/{tstamp}"
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    callbacks = [tb_callback]

    config['fit_config'] = {
        'steps_per_epoch': train_size // config['batch_size'],
        'epochs': config['num_epochs'],
        'callbacks': callbacks,
        'verbose': 1 if config['debug'] else 2,
    }
    config['evaluate_config'] = {
        'steps': valid_size // config['batch_size'],
        'verbose': 0  # there is only 1, 0 possible values
    }
    return config


def multi_node_training(config):
    print("Iniciando ray multi-nodo...")
    ray.init(address='auto', _node_ip_address=environ["ip_head"].split(":")[0],
             _redis_password=environ["redis_password"])
    assert ray.is_initialized(), "ERROR: Ray is not initialized!"
    print('''This cluster consists of
                            {} nodes in total
                            {} CPU resources in total
                            {} GPU resources in total
                        '''.format(len(ray.nodes()), ray.cluster_resources()['CPU'], ray.cluster_resources()['GPU']))

    trainer = TFTrainer(
        model_creator=model_creator,
        data_creator=data_creator,
        num_replicas=config['num_replicas'],
        use_gpu=True,
        verbose=True,
        config=create_config(config)
    )

    trainer.train()  # train with train_config
    eval_stats = trainer.validate()  # validate with validate_config

    ray.shutdown()  # Shutdown ray to prevent an error of ports
    assert not ray.is_initialized(), "ERROR: Ray is still running!"

    return tuple(eval_stats.values())


def distributed_training(config):
    loss, metric = loss_metric(*config['loss_metric'])
    if config['num_replicas'] > 1:
        gpus = tf.config.list_physical_devices('GPU')
        gpus_name = [g.name.split('e:')[1] for g in gpus]
        assert len(gpus_name) == config['num_replicas'], f"Number of GPU available {len(gpus_name)}"
        strategy = tf.distribute.MirroredStrategy(gpus_name[:config['num_replicas']])
        print("Number of GPUs:", strategy.num_replicas_in_sync)  # sanity check
        with strategy.scope():
            model = model_unet_tune(input_shape=(4, 240, 240, 152), filter_start=8, loss_function=loss,
                                    metrics=[metric], initial_learning_rate=config['lr'], amsgrad=config['amsgrad'],
                                    b2=config['b2'], norm=config['norm'])

    else:
        print("Number of GPUs:", 1)
        model = model_unet_tune(input_shape=(4, 240, 240, 152), filter_start=8, loss_function=loss, metrics=[metric],
                                initial_learning_rate=config['lr'], amsgrad=config['amsgrad'], b2=config['b2'],
                                norm=config['norm'])

    # model.summary()
    train_size, valid_size = config['train_size'], config['valid_size']
    steps_per_epoch = train_size // config['batch_size']
    validation_steps = valid_size // config['batch_size']

    files_tr = [config['dataset_path'] + f"train_{i}.tfrecord" for i in range(config['train_shard'])]
    files_val = [config['dataset_path'] + f"valid_{i}.tfrecord" for i in range(config['valid_shard'])]

    train_g = data_generator(files_tr, config, config['train_size'] // 10, train=True)
    valid_g = data_generator(files_val, config, config['valid_size'] // 10, )

    tstamp = "{}".format(datetime.now().strftime("%m_%d_%Y-%H:%M:%S"))
    logdir = f"{config['log_dir']}/logs/{tstamp}"
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    callbacks = [] if config['debug'] else [tb_callback]

    model.fit(x=train_g,
              steps_per_epoch=steps_per_epoch,
              epochs=config['num_epochs'],
              callbacks=callbacks,
              verbose=1 if config['debug'] else 2)

    results = model.evaluate(valid_g, steps=validation_steps, verbose=1 if config['debug'] else 2)
    return tuple(results)  # val_loss, val_metric


def main(args):

    with open(args.config) as f:
        config = json.load(f)

    config['batch_size'] = config['batch_size_per_replica'] * config['num_replicas']
    config['log_dir'] = args.config.split('/config.json')[0]

    config['dataset_path'] = "/gpfs/projects/bsc31/bsc31654/dataset/preprocessed/"
    with open(config['dataset_path'] + "info.json") as f:
        info = json.load(f)
    config.update(info)  # add info to config

    config['prefetch_buffer_size'] = tf.data.experimental.AUTOTUNE
    config['map_num_parallel_calls'] = tf.data.experimental.AUTOTUNE
    config['tfrecord_buffer_size'] = tf.data.experimental.AUTOTUNE
    config['tfrecord_num_parallel_reads'] = tf.data.experimental.AUTOTUNE
    config['loss_metric'] = (dice_loss, dice_coeff)

    multi = config['num_replicas'] > 4

    s = time.time()

    #####################################
    # DEFINE HYPERPARAMETERS TUNE
    #####################################

    lr = [x * config['num_replicas'] for x in [1e-04, 5e-04, 1e-03, 5e-03]]
    norm = ['bn', 'gn']
    amsgrad = [True, False]
    b2 = [0.99, 0.999]

    tune = [lr, norm, b2, amsgrad]
    keys = ['lr', 'norm', 'b2', 'amsgrad']

    ######################################
    ######################################

    best_config = {}
    trials = list(product(*tune))
    for num, trial in enumerate(trials, 1):
        config.update(dict(zip(keys, trial)))
        print(f"Trial {num}/{len(trials)} : {trial}")
        s_ = time.time()
        if multi:
            res = multi_node_training(config)
        else:
            res = distributed_training(config)
        print("Trial time:", time.time() - s_, "s")
        best_config[res] = trial

    best = max(best_config.keys(), key=lambda x: x[1])
    print("Best:", best_config[best])

    print("Elapsed time:", timedelta(seconds=(time.time() - s)), "s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path: Json file configuration")
    _args, _ = parser.parse_known_args()
    assert path.exists(_args.config), f"Config file doesn't exist: {_args.config}"
    main(_args)
