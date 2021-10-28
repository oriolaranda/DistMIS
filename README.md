# Distributing Deep Learning Hyperparameter Tuning for 3D Medical Image Segmentation
Official Repository for the paper: Distributing Deep Learning Hyperparameter Tuning for 3D Medical Image Segmentation 


## Setup
### Installation
Just download the code, and execute inside the directory. You can clone the repository using:
```console
foo@bar:~$ git clone https://github.com/oriolaranda/dist-dl-3d-mis.git
foo@bar:~$ cd dist-dl-3d-mis/
foo@bar:~$ python -m pip install -r requirements.txt
```

### Requirements

The scrips work with python >= 3.7.4 and uses the following packages:
```
tensorflow>=2.3.0
```
### Dataset

The `tfrecord` script assumes that the _Task01_BrainTumor.tar_ has been downloaded from its official webpage [MSD Challenge](http://medicaldecathlon.com/) and extracted to your local machine.

## How To Use
The framework is composed by 4 main scripts: `tfrecord`, `visualize`, `data_parallel` and `exp_parallel`.

### Tfrecord
The `tfrecord` script reads the `dataset.json` file located inside the dataset folder, and creates a tf.data.Dataset from the imagesTr and labelsTr samples. It requires the orignal dataset directory path as argument, e.g. /home/Task01_BrainTumor/. If the target directory is provided, the tf.data.Dataset is serialized and saved in TFRecord format into that directory. Running this script is **the first step** in order to run the other scripts. Serializing the dataset the memory footprint will be larger but it has good benefits:
* Perform offline preprocessing and data augmentation.
* Optimize data reading, online preprocessing and data augmentation.
It can seem the benefits are poor, but in Deep Learning applications with this type of data, these optimization techniques save hours of training.

```console
foo@bar:~$ python tfrecord.py --help
usage: tfrecord.py [-h] --source-dir SOURCE_DIR [--target-dir TARGET_DIR] [--split SPLIT] [--reshape RESHAPE]

optional arguments:
  -h, --help                   Show this help message and exit
  --source-dir SOURCE_DIR      Path: Source data directory. The directory must contain the dataset.json file, 
                               and the two folders imagesTr, labelsTr.
  --target-dir TARGET_DIR      Path: Target data directory. It must exist. It is where the TFRecord data will be 
                               written to.
  --split SPLIT                Tuple: Ratios into which the dataset will be divided (all sum up to 1). Train, 
                               validation and test set. Default=(0.7, 0.15, 0.15).
  --reshape RESHAPE            Tuple: Shape of the written data. Default=(240, 240, 152) which is its original 
                               shape. If a different shape is provided, a trilinear resize will be applied to 
                               the data.
```
#### Examples:
Creating the target directory
```console
foo@bar:~$ mkdir dataset
```
Creating the tfrecord dataset into the created directory _dataset_.
```console
foo@bar:~$ python tfrecord.py --source-dir /home/Task01_BrainTumor/ --target-source /home/dataset/
```
Creating a tfrecord dataset with smaller size data, and different split sets.

```console
foo@bar:~$ python tfrecord.py --source-dir /home/Task01_BrainTumor/ --target-source /home/dataset/ \ --reshape (120, 120, 152) --split (0.8, 0.1, 0.1)
```

### Visualize
The `visualize` script is just an auxiliary script for visualizing the data after doing the tfrecord
and other possible transformations, e.g. offline data_augmentation. It is also useful for debugging
purposes, e.g. testing some transformation or preprocessing functions, before deploying.

```console
usage: visualize.py [-h] --dataset-dir DATASET_DIR [--sample SAMPLE] [--data-shape DATA_SHAPE] 
                         [--no-screen NO_SCREEN]

optional arguments:
  -h, --help                    Show this help message and exit
  --dataset-dir DATASET_DIR     Path: TFRecord dataset directory.
  --sample SAMPLE               Int: Sample to visualize. Default=0. It has to be 
                                0 <= sample <= size_dataset.
  --data-shape DATA_SHAPE       Tuple: Shape of the data in the dataset path provided. 
                                Default=(240, 240, 152) which is the orginal data shape.
  --no-screen NO_SCREEN         Bool: No X session (graphical Linux desktop) mode. Default=False. 
                                If set to True, a GIF file will be saved in the current directory ('./') 
                                containing the plotted image.

```
#### Examples:
Visualizing the written data in TFRecord format, the default sample is the number 0.
```console
foo@bar:~$ python visualize.py --dataset-dir /home/dataset/
```
Visualizing the sample number 350, since we are working via ssh with no x session, we enable _--no-screeen_ flag to save a GIF file.
```console
foo@bar:~$ python visualize.py --dataset-dir /home/dataset/ --sample 350 --no-screen True
```
![](./images/giphy.gif)

### Data Parallelism
The `data_parallel` script is the first approach presented in the paper, given a model in tensorflow and a TFRecord dataset it performs data parallelism. 
Data parallelism consists in, given n GPUs, the model is replicated n times and each replica is sent to a GPU. After that, the data is split into n chunks, i.e. the batch size is diveded by n, these chunks are distributed across the GPUs, where each chunk is assigned to a GPU. If we are training m models and m >= n, then we proceed sequentially for each model. Since we are using more than 1 GPU we are speeding-up the training of each model, and therefore, the m models.
Our cluster has 4 GPUs per node, so if the number of GPUs used is less than 4, i.e. we are using only one node, tf.MirroredStrategy is used. For multi-node, i.e. >= 4 GPUs, we use ray.cluster which handles all the comunications between nodes and ray.sgd which is a wrapper around tf.MultiWorkerMirroredStrategy.
Both tf.MirroredStrategy and tf.MultiWorkerMirroredStrategy are built-in functions from tensorflow distributed API.

### Experiment Parallelism
The `exp_parallel` script is the second approach presented in the paper, given a model in tensorflow and a TFRecord dataset it performs experiment parallelism using ray.tune which manages all the low level parallelism implementaion.
Experiment parallelism consists in, given n GPUs, m models and m >= n, assigning a model to each GPU available. Hence, we are training n models at the same time, speeding-up the computations of all the m models.
As mentioned above our cluster has 4 GPU per node, so if the number of GPUs used is less than 4, i.e. we are using only one node, ray.tune is used. For multi-node, i.e. >= 4 GPUs, we use ray.cluster which handles all the comunications between nodes and ray.tune.



