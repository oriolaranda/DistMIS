# Distributing Deep Learning Hyperparameter Tuning for 3D Medical Image Segmentation
Official Repository for the paper: Distributing Deep Learning Hyperparameter Tuning for 3D Medical Image Segmentation 


## Setup
### Requirements

The scrips work with python >= 3.7.4 and uses the following packages:
```
tensorflow>=2.3.0
```
### Dataset

The `tfrecord` script assumes that the _Task01_BrainTumor.tar_ has been downloaded from its official webpage [MSD Challenge]{http://medicaldecathlon.com/} and extracted to your local machine.

## How To Use

### Tfrecord
The tfrecord script reads the `dataset.json` file located inside the dataset folder, and creates a tf.data.Dataset from the imagesTr and labelsTr samples. It requires the dataset folder path as argument, e.g. /home/Task01_BrainTumor/.

```console
foo@bar:~$ python tfrecord.py --help
usage: tfrecord.py [-h] --source_dir SOURCE_DIR [--target_dir TARGET_DIR] [--split SPLIT]

optional arguments:
  -h, --help                   Show this help message and exit
  --source_dir SOURCE_DIR      Path: Source data directory. The directory must contain the dataset.json file, 
                               and the two folders imagesTr, labelsTr.
  --target_dir TARGET_DIR      Path: Target data directory. It must exist. It is where the TFRecord data will be 
                               written to.
  --split SPLIT                Tuple: Ratios into which the dataset will be divided (all sum up to 1). Train, 
                               validation and test set. Default=(0.7, 0.15, 0.15).
  --reshape RESHAPE            Tuple: Shape of the written data. Default=(240, 240, 152) which is its original 
                               shape. If a different shape is provided, a trilinear resize will be applied to 
                               the data.
```
#### Example:
Creating the target directory
```console
foo@bar:~$ mkdir dataset
```
Creating the tfrecord dataset into the created directory _dataset_
```console
foo@bar:~$ python tfrecord.py --source_dir /home/Task01_BrainTumor/ --target_source /home/dataset/
```
Creating a tfrecord dataset with smaller size data, and different split sets

```console
foo@bar:~$ python tfrecord.py --source_dir /home/Task01_BrainTumor/ --target_source /home/dataset/ \ 
--reshape (120, 120, 152) --split (0.8, 0.1, 0.1)
```

### Visualize
The `visualize` script is just an auxiliary script for visualizing the data after doing the tfrecord
and other possible transformations, e.g. offline data_augmentation. It is also useful for debugging
purposes, e.g. testing some transformation or preprocessing functions, before deploying.

```console
usage: visualize.py [-h] --dataset_dir DATASET_DIR [--sample SAMPLE] [--data_shape DATA_SHAPE] 
                         [--no-screen NO_SCREEN]

optional arguments:
  -h, --help                    Show this help message and exit
  --dataset_dir DATASET_DIR     Path: TFRecord dataset directory.
  --sample SAMPLE               Int: Sample to visualize. Default=0. It has to be 
                                0 <= sample <= size_dataset.
  --data_shape DATA_SHAPE       Tuple: Shape of the data in the dataset path provided. 
                                Default=(240, 240, 152) which is the orginal data shape.
  --no-screen NO_SCREEN         Bool: No X session (graphical Linux desktop) mode. Default=False. 
                                If set to True, a GIF file will be saved in the current directory ('./') 
                                containing the plotted image.

```
#### Example:
Visualizing the written data in TFRecord format, the default sample is the number 0.
```console
foo@bar:~$ python visualize.py --dataset_dir /home/dataset/
```

### Data Parallelism


### Experiment Parallelism

