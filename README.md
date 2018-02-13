# v1_predictor
Training and evaluating the CNN for "Using deep learning to reveal the neural code for images in primary visual cortex" by William F. Kindel, Elijah D. Christensen and Joel Zylberberg https://arxiv.org/abs/1706.06208 (2017).

## Overview

This program trains and evaluates a convolutional neural network (CNN) whose input is an image and whose output is the predicted firing rates of every neuron in a given data file. This CNN has two convolutional layers followed by a densily connected hidden layer with hyper-parameters described below. To quantify the performance of the predictor, we compare the network’s predicted firing rates to the neurons’ measured firing rates using a held-out evaluation set using  the Pearson correlation coefficient.

## Versions

- Python 3.5.2
- Tensorlow 1.0.1
> Also see requirements.txt

## Necessary Files

### python files
- runnet.py
- buildnet.py

### data files
stored in data/data.tar.gz

```bash
$ tar -xvzf data.tar.gz

data/
├── 01mean50ms_smallim_d2_crop.mat
├── 02mean50ms_smallim_d2_crop.mat
├── 03mean50ms_smallim_d2_crop.mat
├── 04mean50ms_smallim_d2_crop.mat
├── 05mean50ms_smallim_d2_crop.mat
├── 06mean50ms_smallim_d2_crop.mat
├── 07mean50ms_smallim_d2_crop.mat
├── 08mean50ms_smallim_d2_crop.mat
├── 09mean50ms_smallim_d2_crop.mat
└── 10mean50ms_smallim_d2_crop.mat
```

## Examples

### Building and Running the network

From a Unix terminal:

```bash
$ python3 runnet.py
```

### Plotting Example

For example of plotting output files run:
- examples/plot1516647610.py

```bash
$ cd examples
$ python3 plot1516647610.py
```

Which produces:
- examples/training_manualsave_1516647610.npy
- examples/persondata1516647610.npy

## Network Outputs

Written to the manualsave folder located in the working directory. Folder created if not present.

All output files have the same ID siffix (denoted UNIXTIME) which is the Unix time when the training starts.

Outputs:
- training_plot_UNIXTIME.png -- a plot of the training progress writting as the training goes on
- training_manualsave_UNIXTIME.npy -- the data that does into the training plot. 
    Load as: [FLAGS, trainlist, earlystoplist, evallist, rlist, step, lossbaseline, traintrials, earlystoptrials, evaltrials] = np.load(training_manualsave_UNIXTIME.npy)
- personplotUNIXTIME.png -- a bar plot of the Pearson correlation coefficient over all neurons for the best nextowrk updated as the training goes on.  
- persondataUNIXTIME.npy -- the data that does into the Pearson plot.
    Load as: [rval,step,traintrials, earlystoptrials, evaltrials, FLAGS] = np.load(persondataUNIXTIME.npy)
- network_manualsave_UNIXTIME.npy -- the trained weights/parameters of the CNN for the best nextowrk updated as the training goes on.
    Load as: [WC1, BC1, WC2, BC2, WH3, BH3, WL4, BL4, step] = np.load(network_manualsave_UNIXTIME.npy)

# CNN Hyperparameters

The hyper-parameters of the are written as a FLAG which can be adjusted.

```bash
$ python3 runnet.py --help
usage: runnet.py [-h] [--learning_rate LEARNING_RATE] [--max_steps MAX_STEPS]
                 [--conv1 CONV1] [--conv2 CONV2] [--hidden1 HIDDEN1]
                 [--hidden2 HIDDEN2] [--batch_size BATCH_SIZE]
                 [--trainingset_frac TRAININGSET_FRAC]
                 [--earlystop_frac EARLYSTOP_FRAC] [--dropout DROPOUT]
                 [--save SAVE] [--savetraining SAVETRAINING]
                 [--savenetwork SAVENETWORK] [--conv1size CONV1SIZE]
                 [--nk1 NK1] [--nstride1 NSTRIDE1] [--conv2size CONV2SIZE]
                 [--nk2 NK2] [--nstride2 NSTRIDE2] [--fileindex FILEINDEX]

optional arguments:
  -h, --help            show this help message and exit
  --learning_rate LEARNING_RATE
                        Initial learning rate.
  --max_steps MAX_STEPS
                        Number of steps to run trainer.
  --conv1 CONV1         Number of filters in conv 1.
  --conv2 CONV2         Number of filters in conv 2.
  --hidden1 HIDDEN1     Number of units in hidden layer 1.
  --hidden2 HIDDEN2     Number of units in hidden layer 2. Not used.
  --batch_size BATCH_SIZE
                        Batch size.
  --trainingset_frac TRAININGSET_FRAC
                        Training set size (fraction of images).
  --earlystop_frac EARLYSTOP_FRAC
                        Early stop set size (fraction of images).
  --dropout DROPOUT     ...
  --save SAVE           If true, save the results.
  --savetraining SAVETRAINING
                        If true, save the traing.
  --savenetwork SAVENETWORK
                        If true, save the network
  --conv1size CONV1SIZE
                        Size (linear) of convolution kernel larer 1.
  --nk1 NK1             Size of max pool kernel layer 1.
  --nstride1 NSTRIDE1   Size of max pool stride layer 1.
  --conv2size CONV2SIZE
                        Size (linear) of convolution kernel larer 2.
  --nk2 NK2             Size of max pool kernel layer 2.
  --nstride2 NSTRIDE2   Size of max pool stride.
  --fileindex FILEINDEX
                        index for which file to load
```

- FLAGS.learning_rate -- the learning rate
- FLAGS.max_steps -- maximum number of steps to run trainer
- FLAGS.dropout -- drop out keep rate during training
- FLAGS.batch_size -- batch size for training
- FLAGS.trainingset_frac -- training set size (fraction of images)
- FLAGS.earlystop_frac -- early stop set size (fraction of images)
- FLAGS.conv1 -- number of filters in conv 1.
- FLAGS.conv2 -- number of filters in conv 2.
- FLAGS.hidden1 -- number of units in hidden layer 1
- FLAGS.conv1size -- size (linear) of convolution kernel larer 1
- FLAGS.nk1 -- size of max pool kernel layer 1.
- FLAGS.nstride1 -- size of max pool stride layer 1.
- FLAGS.conv2size -- size (linear) of convolution kernel larer 2
- FLAGS.nk2 -- size of max pool kernel layer 2
- FLAGS.nstride2 -- size of max pool stride for conv 2
- FLAGS.fileindex -- intiger index for which file to load
- FLAGS.save -- if true, write output files
