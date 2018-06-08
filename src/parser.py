import os
import argparse

def gen_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--data_dir',
      type=str,
      default=os.path.join(os.getcwd(),'data'),
      help='Data directory containing mat files of the format: 0Nmean50ms_smallim_d2_crop'
    )
    parser.add_argument(
      '--save_dir',
      type=str,
      default=os.path.join(os.getcwd(),'manualsave'),
      help='Directory to save outputs'
    )
    parser.add_argument(
      '--save',
      default = True,
      help='If true, save the results.'
    )
    parser.add_argument(
      '--savetraining',
      default = True,
      help='If true, save the traing.'
    )
    parser.add_argument(
      '--savenetwork',
      default = False,
      help='If true, save the network'
    )
    parser.add_argument(
      '--fileindex',
      type=int,
      default=1,
      help='index for which file to load'
    )

    # Training / Optimizing params
    training = parser.add_argument_group('training')
    training.add_argument(
      '--learning_rate',
      type=float,
      default=1.00e-4,
      help='Initial learning rate. Default=1.00e-4'
    )
    training.add_argument(
      '--max_steps',
      type=int,
      default=60000,
      help='Number of steps to run trainer.'
    )
    training.add_argument(
      '--batch_size',
      type=int,
      default=50,
      help='Batch size. '
    )
    training.add_argument(
      '--trainingset_frac',
      type=float,
      default=2/3,
      help='Training set size (fraction of images).'
    )
    training.add_argument(
      '--earlystop_frac',
      type=float,
      default=1/7,
      help='Early stop set size (fraction of images).'
    )

    # Define Convolutional parameters
    cnn = parser.add_argument_group('CNN params')
    cnn.add_argument(
      '--conv1',
      type=int,
      default=16,
      help='Number of filters in conv 1.'
    )
    cnn.add_argument(
      '--conv2',
      type=int,
      default=32,
      help='Number of filters in conv 2.'
    )
    cnn.add_argument(
      '--conv1size',
      type=int,
      default=3,
      help='Size (linear) of convolution kernel larer 1.'
    )
    cnn.add_argument(
      '--nk1',
      type=int,
      default=3,
      help='Size of max pool kernel layer 1.'
    )
    cnn.add_argument(
      '--nstride1',
      type=int,
      default=2,
      help='Size of max pool stride layer 1.'
    )
    cnn.add_argument(
      '--conv2size',
      type=int,
      default=3,
      help='Size (linear) of convolution kernel larer 2.'
    )
    cnn.add_argument(
      '--nk2',
      type=int,
      default=3,
      help='Size of max pool kernel layer 2.'
    )
    cnn.add_argument(
      '--nstride2',
      type=int,
      default=2,
      help='Size of max pool stride.'
    )
    cnn.add_argument(
      '--numconvlayer',
      type=int,
      default=2,
      help='number of convolutional layers'
    )

    hidden = parser.add_argument_group('hidden layer')
    hidden.add_argument(
      '--hidden1',
      type=int,
      default=300,
      help='Number of units in hidden layer 1.'
    )
    hidden.add_argument(
      '--hidden2',
      type=int,
      default=1,
      help='Number of units in hidden layer 2. Not used.'
    )
    parser.add_argument(
      '--dropout',
      type=float,
      default=0.65,
      help='...'
    )

    return parser
