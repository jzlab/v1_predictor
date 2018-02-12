Training and evaluating the CNN for "Using deep learning to reveal the neural code for images in primary visual cortex" by William F. Kindel, Elijah D. Christensen and Joel Zylberberg https://arxiv.org/abs/1706.06208 (2017).

---- Overview ---

This program trains and evaluates a convolutional neural network (CNN) whose input is an image and whose output is the predicted firing rates of every neuron in a given data file. This CNN has two convolutional layers followed by a densily connected hidden layer with hyper-parameters described below. To quantify the performance of the predictor, we compare the network’s predicted firing rates to the neurons’ measured firing rates using a held-out evaluation set using  the Pearson correlation coefficient.


---- Run ---

From a Unix terminal:
$ python3 runnet.py

For exampel of ploting output files run:
$ python3 plot1516647610.py


---- Files needed ----

python files:
runnet.py
buildnet.py
data files:
01mean50ms_smallim_d2_crop.mat
02mean50ms_smallim_d2_crop.mat
03mean50ms_smallim_d2_crop.mat
04mean50ms_smallim_d2_crop.mat
05mean50ms_smallim_d2_crop.mat
06mean50ms_smallim_d2_crop.mat
07mean50ms_smallim_d2_crop.mat
08mean50ms_smallim_d2_crop.mat
09mean50ms_smallim_d2_crop.mat
10mean50ms_smallim_d2_crop.mat

plotting example:
plot1516647610.py
training_manualsave_1516647610.npy
persondata1516647610.npy


---- Versions -----

Python 3.5.2
Tensorlow 1.0.1


---- Network outputs -----

Written to the manualsave folder located in the working directory. Folder created if not present.

All output files have the same ID siffix (denoted UNIXTIME) which is the Unix time when the training starts.

Outputs:
training_plot_UNIXTIME.png -- a plot of the training progress writting as the training goes on
training_manualsave_UNIXTIME.npy -- the data that does into the training plot. 
    Load as: [FLAGS, trainlist, earlystoplist, evallist, rlist, step, lossbaseline, traintrials, earlystoptrials, evaltrials] = np.load(training_manualsave_UNIXTIME.npy)
personplotUNIXTIME.png -- a bar plot of the Pearson correlation coefficient over all neurons for the best nextowrk updated as the training goes on.  
persondataUNIXTIME.npy -- the data that does into the Pearson plot.
    Load as: [rval,step,traintrials, earlystoptrials, evaltrials, FLAGS] = np.load(persondataUNIXTIME.npy)
network_manualsave_UNIXTIME.npy -- the trained weights/parameters of the CNN for the best nextowrk updated as the training goes on.
    Load as: [WC1, BC1, WC2, BC2, WH3, BH3, WL4, BL4, step] = np.load(network_manualsave_UNIXTIME.npy)


--- CNN hyper-parameters ----

The hyper-parameters of the are written as a FLAG which can be adjusted.

FLAGS.learning_rate -- the learning rate
FLAGS.max_steps -- maximum number of steps to run trainer
FLAGS.dropout -- drop out keep rate during training
FLAGS.batch_size -- batch size for training
FLAGS.trainingset_frac -- training set size (fraction of images)
FLAGS.earlystop_frac -- early stop set size (fraction of images)
FLAGS.conv1 -- number of filters in conv 1.
FLAGS.conv2 -- number of filters in conv 2.
FLAGS.hidden1 -- number of units in hidden layer 1
FLAGS.conv1size -- size (linear) of convolution kernel larer 1
FLAGS.nk1 -- size of max pool kernel layer 1.
FLAGS.nstride1 -- size of max pool stride layer 1.
FLAGS.conv2size -- size (linear) of convolution kernel larer 2
FLAGS.nk2 -- size of max pool kernel layer 2
FLAGS.nstride2 -- size of max pool stride for conv 2
FLAGS.fileindex -- intiger index for which file to load 
FLAGS.save -- if true, write output files
