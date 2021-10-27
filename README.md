# Distributing Deep Learning Hyperparameter Tuning for 3D Medical Image Segmentation
Official Repository for the paper: Distributing Deep Learning Hyperparameter Tuning for 3D Medical Image Segmentation 


## Setup
### Requirements

The scrips work with python >= 3.7.4 and uses the following packages:
```
tensorflow>=2.3.0
```
### Dataset

The `tfrecord` script assumes that the Task01_BrainTumor.tar has been downloaded from its official webpage [MSD Challenge]{http://medicaldecathlon.com/} and extracted to your local machine.

## How To Use

### Tfrecord.py
The tfrecord script reads the `dataset.json` file located inside the dataset folder, and creates a tf.data.Dataset from the imagesTr and labelsTr samples. It requires the dataset folder path as argument, e.g. /home/Task01_BrainTumor/.

```console
foo@bar:~$ python tfrecord.py --help
usage: tfrecord.py [-h] --source_dir SOURCE_DIR [--target_dir TARGET_DIR] [--split SPLIT]

optional arguments:
  -h, --help                   Show this help message and exit
  --source_dir SOURCE_DIR      Source data directory. The directory must contain the dataset.json file, 
                               and the two folders imagesTr, labelsTr
  --target_dir TARGET_DIR      Target data directory. It must exist. Where the TFRecord data will be saved.
  --split SPLIT                Ratios into which the dataset will be divided. Train, validation and test set. 
                               Default (0.7, 0.15, 0.15).
```
Example:
```console
foo@bar:~$ mkdir dataset
foo@bar:~$ python tfrecord.py --source_dir /home/Task01_BrainTumor/ --target_source /home/dataset/
