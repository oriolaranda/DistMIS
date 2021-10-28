########################################################################################################################
# @author Oriol Aranda (https://github.com/oriolaranda/)
# @date Oct 2021
########################################################################################################################

import time
import argparse
from datetime import datetime, timedelta
import json
from os import path, environ
import tensorflow as tf
import ray
from ray import tune
from ray.tune import CLIReporter
from utils import dice_loss, dice_coeff, TuneReporter, loss_metric, data_generator
from model import model_unet_tune


def custom_train_model(config):
    loss, metric = loss_metric(*config['loss_metric'])
    print("Using number of GPUs:", 1)
    model = model_unet_tune(input_shape=(4, *config['input_shape']), filter_start=8,
                            loss_function=loss, metrics=[metric], initial_learning_rate=config['lr'],
                            amsgrad=config['amsgrad'], k=3, b2=config['b2'], norm=config['norm'])

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
    callbacks += [TuneReporter()]

    model.fit(x=train_g,
              steps_per_epoch=steps_per_epoch,
              epochs=config['num_epochs'],
              verbose=1 if config['debug'] else 2)

    model.evaluate(valid_g, steps=validation_steps, callbacks=[TuneReporter()],
                   verbose=1 if config['debug'] else 2)


def main(args):
    with open(args.config) as f:
        config = json.load(f)

    config['num_replicas'] = 1
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

    multi = config.get('multi', False)

    s = time.time()
    if multi:
        ray.init(address='auto', _node_ip_address=environ["ip_head"].split(":")[0],
                 _redis_password=environ["redis_password"])
        print('''This cluster consists of
                {} nodes in total
                {} CPU resources in total
                {} GPU resources in total
            '''.format(len(ray.nodes()), ray.cluster_resources()['CPU'], ray.cluster_resources()['GPU']))

    else:
        print("Iniciando ray...")
        ray.init(address='auto', _redis_password='5241590000000000')
        print("Ray iniciado!")

    ###########################################################################
    # DEFINE HYPERPARAMETERS TO TUNE
    ###########################################################################

    config['lr'] = tune.grid_search([1e-04, 5e-04, 1e-03, 5e-03])
    config['norm'] = tune.grid_search(['bn', 'gn'])
    config['b2'] = tune.grid_search([0.99, 0.999])
    config['amsgrad'] = tune.grid_search([True, False])

    ###########################################################################
    ###########################################################################

    reporter = CLIReporter(metric_columns=['val_dice_coefficient', 'time_total_s', 'training_iteration'])

    analysis = tune.run(
        custom_train_model,
        name=config['log_dir'].split('/')[-1],
        metric="val_dice_coefficient",
        mode="max",
        num_samples=1,
        resources_per_trial={
            "cpu": 10,
            "gpu": 1
        },
        verbose=3,
        progress_reporter=reporter,
        config=config)
    print("Best:", analysis.best_config)

    print("Elapsed time:", timedelta(seconds=(time.time() - s)), "s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path: Json file configuration")
    _args, _ = parser.parse_known_args()
    assert path.exists(_args.config), f"Config file doesn't exist: {_args.config}"
    main(_args)
