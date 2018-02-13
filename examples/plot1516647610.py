"""Exmple for loading and plotting training data and results"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
import argparse
import os
import os.path
import sys
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy
from scipy.stats import pearsonr


# setting up saving and file IDs
IDtag = '1516647610'
working_dir = os.getcwd()
print("Working dir: " + working_dir)
#mansave_dir = working_dir + '/manualsave'
mansave_dir = working_dir 
savedirname = mansave_dir + '/training_manualsave_' + IDtag + '.npy'
savenetworkname = mansave_dir + '/network_manualsave_' + IDtag + '.npy'
pearsonsavedatadir = mansave_dir + '/persondata' + IDtag + '.npy'


def plotpearson():
	[rval,step,traintrials, earlystoptrials, evaltrials, FLAGS] = np.load(pearsonsavedatadir)
	figN, axN = plt.subplots()
	bar_width = 0.40
	xvals = np.arange(len(rval))+1-bar_width/2
	axN.bar(xvals, rval)
	axN.set_xlabel('cell', color='k')
	axN.set_ylabel('r', color='k')
	plt.show()


def load_network():
	[WC1, BC1, WC2, BC2, WH3, BH3, WL4, BL4, step] = np.load(savenetworkname)


def plottraining():
	[FLAGS, trainlist, earlystoplist, evallist, rlist, step, lossbaseline, traintrials, earlystoptrials, evaltrials] = np.load(savedirname)

	minindex = np.argmin(np.asarray(earlystoplist))
	xrange = max(step)
    
	textx = 100 + xrange/8
	if (textx<step[minindex]) & (step[minindex]<xrange/2):
		textx = xrange/12 + step[minindex]
	ymax = 2.0*lossbaseline
	ymin = -.1*lossbaseline
	deltay = ymax-ymin
	figA, axA = plt.subplots()

	axA.plot(np.asarray(step), np.asarray(trainlist), 'r', label='training')
	axA.plot(np.asarray(step), np.asarray(earlystoplist), 'r--', label='early stop')
	axA.plot(np.asarray(step), np.asarray(evallist), 'r:', label='evaluation')
	axA.set_ylim([ymin,ymax])
	axA.set_xlabel('step')
  	## Make the y-axis label, ticks and tick labels match the line color.
  	#axA.axvline(x=step[minindex], ymin=-.1, ymax = 2, linewidth=2, color='k')
	axA.set_ylabel('least squared loss', color='k')
	axA.tick_params('y', colors='k')
  	#axA.annotate('Step: %d (frac noise:%.3f ) ' % (step[minindex],fracnoiseexplane[minindex] ), xy=(textx, 0.95*deltay+ymin))
  	#axA.annotate('Max pool 1 stide: %d ' % (FLAGS.nstride), xy=(textx, 0.90*deltay+ymin))
  	#axA.annotate('Max pool 1 kernel size: %d ' % (FLAGS.nk), xy=(textx, 0.85*deltay+ymin))
  	#axA.annotate('Conv 1 size: %d X %d ' % (FLAGS.conv1size, FLAGS.conv1size), xy=(textx, 0.80*deltay+ymin))
  	#axA.annotate('number of L 1 filters: %d' % (FLAGS.conv1), xy=(textx, 0.75*deltay+ymin))
	#axA.annotate('Max pool 2 stide: %d ' % (FLAGS.nstride2), xy=(textx, 0.70*deltay+ymin))
  	#axA.annotate('Max pool 2 kernel size: %d ' % (FLAGS.nk2), xy=(textx, 0.65*deltay+ymin))
  	#axA.annotate('Conv 2 size: %d X %d ' % (FLAGS.conv2size, FLAGS.conv2size), xy=(textx, 0.60*deltay+ymin))
  	#axA.annotate('number of L 1 filters: %d' % (FLAGS.conv2), xy=(textx, 0.55*deltay+ymin))
  	#axA.annotate('Dropout keep rate: %.3f ' % (FLAGS.dropout), xy=(textx, 0.50*deltay+ymin))
 	#axA.annotate('%d hidden elements' % (FLAGS.hidden1 ), xy=(textx, 0.45*deltay+ymin))
	#axA.annotate('Learning rate: %.1e' % (FLAGS.learning_rate), xy=(textx, 0.40*deltay+ymin))
	plt.legend(loc=4)
	axB = axA.twinx()
	axB.plot(np.asarray(step), np.asarray(rlist), 'k')
	axB.set_ylim([-0.1,1])
	axB.set_ylabel('r', color='k')
	axB.tick_params('y', colors='k')
	figA.tight_layout()
	plt.show()


def main(_):
  plottraining()
  plotpearson()

# run main 
tf.app.run(main=main, argv=[sys.argv[0]])
