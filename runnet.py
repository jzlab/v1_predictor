#
#
#
#

"""Trains and Evaluates a CNN on processed Kohn Monkey data"""
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

FLAGS = None
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=2.00e-4,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--max_steps',
      type=int,
      default=60000,
      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--conv1',
      type=int,
      default=16,
      help='Number of filters in conv 1.'
  )
  parser.add_argument(
      '--conv2',
      type=int,
      default=32,
      help='Number of filters in conv 2.'
  )
  parser.add_argument(
      '--hidden1',
      type=int,
      default=300,
      help='Number of units in hidden layer 1.'
  )
  parser.add_argument(
      '--hidden2',
      type=int,
      default=1,
      help='Number of units in hidden layer 2. Not used.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=50,
      help='Batch size. '
  )
  parser.add_argument(
      '--trainingset_frac',
      type=float,
      default=2/3,
      help='Training set size (fraction of images).'
  )
  parser.add_argument(
      '--earlystop_frac',
      type=float,
      default=1/7,
      help='Early stop set size (fraction of images).'
  )
  parser.add_argument(
      '--dropout',
      type=float,
      default=0.55,
      help='...'
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
      '--conv1size',
      type=int,
      default=7,
      help='Size (linear) of convolution kernel larer 1.'
  )
  parser.add_argument(
      '--nk1',
      type=int,
      default=3,
      help='Size of max pool kernel layer 1.'
  )
  parser.add_argument(
      '--nstride1',
      type=int,
      default=2,
      help='Size of max pool stride layer 1.'
  )
  parser.add_argument(
      '--conv2size',
      type=int,
      default=7,
      help='Size (linear) of convolution kernel larer 2.'
  )
  parser.add_argument(
      '--nk2',
      type=int,
      default=3,
      help='Size of max pool kernel layer 2.'
  )
  parser.add_argument(
      '--nstride2',
      type=int,
      default=2,
      help='Size of max pool stride.'
  )
  parser.add_argument(
      '--fileindex',
      type=int,
      default=1,
      help='index for which file to load'
  )


FLAGS, unparsed = parser.parse_known_args()

# setting up saving and file IDs
IDtag = str(int(time.time())) # use a unix time ID for each training
working_dir = os.getcwd()
print("Working dir: " + working_dir)
#mansave_dir = '/home/jzlab1/Dropbox/kohn/NIPSfullSetSmImg/twoconv7/scaledloss50ms/01/mansave'
mansave_dir = working_dir + '/manualsave' # the directory to save to
print("manual save dir: " + mansave_dir)
savedirname = mansave_dir + '/training_manualsave_' + IDtag #filename for saving training
savenetworkname = mansave_dir + '/network_manualsave_' + IDtag # saving network parameters
savefigname = mansave_dir + '/training_plot_' + IDtag # saving traing figure
neuronbessavedirt = mansave_dir + '/nueronloss_' + IDtag # not used 
pearsonsavedir = mansave_dir + '/personplot' + IDtag # saving performance figure 
pearsonsavedatadir = mansave_dir + '/persondata' + IDtag # saving performance data 
losssavedir = mansave_dir + '/lossplot' + IDtag # not used

#making a folder to save to 
if (tf.gfile.Exists(mansave_dir) == 0):
	tf.gfile.MakeDirs(mansave_dir)

#import the network
import buildnet
from buildnet import ConvNet 
#from buildnet import simpleRNN
#from buildnet import RConvNet
#from buildnet import LnonL


class loaddata(object):
	
	def __init__(self, filenamelist):	
		## load the data from files in 'filenamelist'
		## creats an object with trianing, early stop and evaluation formatted data 
		## update filenames in "filenamelist" and the argument in " = mat_contents['argument']"
		numfiles = len(filenamelist)
		print("loading %d file(s)" % (numfiles))
		filename = filenamelist[0]
		mat_contents = sio.loadmat(filename)
		activity = mat_contents['activity'] # the raw outputs (firing rates of nuerons), update as needed
		# activity data formated as array with: [number of images, number of nuerons]  
		images = mat_contents['images'] # the raw inputs (image), update as needed 
		# images data formated as: [number of image, number or x pixles, number of y pixles].
		# for black and white images 
		activitydum = activity
		ncellsdum = np.shape(activitydum)[1]
		print([filename + " ncells:"])
		print(ncellsdum)
		if numfiles > 0: # for concatinating servaral files. For adding new outputs (nuerons) for same inputs (image)
			for index in range(1, numfiles):
				filename = filenamelist[index]
				mat_contents = sio.loadmat(filename)
				activitydum = mat_contents['activity']
				activity = np.append(activity,activitydum, axis = 1)
				ncellsdum = np.shape(activitydum)[1]
				print([filename + " ncells:"])
				print(ncellsdum) 
		
		
		## parameters of data  
		actdatashape = activity.shape
		imgdatashape = images.shape
		numtrials = actdatashape[0]
		numimg = imgdatashape[0]
		numpixx = imgdatashape[1]
		numpixy = imgdatashape[2]
		numcell = actdatashape[1]
		
		## format into sets
		randnumimg = np.random.permutation(numtrials)
		numtrain = int(np.ceil(numimg*FLAGS.trainingset_frac))
		numerlystop = int(np.floor(numimg*FLAGS.earlystop_frac))
		numeval =  numimg - numtrain - numerlystop
		traintrials = randnumimg[0:numtrain]
		evaltrials = randnumimg[numtrain:numtrain+numeval]
		earlystoptrials = randnumimg[numtrain+numeval:numimg+1]
		x = images
		y = activity
		xtrain = np.reshape(x[traintrials,:,:],(numtrain,numpixx,numpixy,1))
		xstop = np.reshape(x[earlystoptrials,:,:],(numerlystop,numpixx,numpixy,1))
		xeval = np.reshape(x[evaltrials,:,:],(numeval,numpixx,numpixy,1))
		ytrain = y[traintrials,:]
		ystop = y[earlystoptrials,:]
		yeval = y[evaltrials,:]
		 
		## set the outputs
		self.activity = activity
		self.images = images 
		self.xtrain = xtrain
		self.xstop = xstop
		self.xeval = xeval
		self.ytrain = ytrain
		self.ystop = ystop
		self.yeval = yeval
		self.numcell = numcell
		self.numtrials = numtrials
		self.numimg = numimg
		self.numpixx = numpixx
		self.numpixy = numpixy
		self.traintrials = traintrials
		self.numtrain = numtrain
		self.evaltrials = evaltrials
		self.numeval = numeval
		self.earlystoptrials = earlystoptrials
		self.numerlystop = numerlystop

# list of filenames for data. update this.
#files must be in same dir
filearray = ['01mean50ms_smallim_d2_crop.mat', '03mean50ms_smallim_d2_crop.mat', '04mean50ms_smallim_d2_crop.mat', 
				'05mean50ms_smallim_d2_crop.mat', '06mean50ms_smallim_d2_crop.mat', '07mean50ms_smallim_d2_crop.mat', 
				'08mean50ms_smallim_d2_crop.mat', '09mean50ms_smallim_d2_crop.mat', '10mean50ms_smallim_d2_crop.mat']
fileindex = FLAGS.fileindex
data = loaddata([filearray[fileindex]]) # loads and formats all the data
print("total number of cells is %d" % (data.numcell))

print("y eval shape")
print(np.shape(data.yeval))
evalvar = np.var(data.yeval,axis=0)
print("eval var is") 
print(np.mean(evalvar))


numtrain = data.numtrain
numerlystop = data.numerlystop
numeval =  data.numeval 
numbatch = np.minimum(numtrain,FLAGS.batch_size)
numtrials = data.numtrials
numpixx = data.numpixx  
numpixy = data.numpixy
print('number of trials %d' % (numtrials))
print('number of  pixels are %d X %d' % (numpixx,numpixy))


def run_training(lossbaseline):
	#start the training
	with tf.Graph().as_default():
		# generate placeholders
		images_placeholder = tf.placeholder(tf.float32, shape=(None, data.numpixx, data.numpixy, 1))
		activity_placeholder = tf.placeholder(tf.float32, shape=(None,data.numcell))
		keep_prob_placeholder = tf.placeholder(tf.float32)
		baselineloss_placeholder = tf.placeholder(tf.float32, shape=(data.numcell))

		# network hyper-parameters as arrays
		#2 conv layers
		num_filter_list = [FLAGS.conv1, FLAGS.conv2] # [16,32]
		filter_size_list = [FLAGS.conv1size, FLAGS.conv2size] # [7,7]
		pool_stride_list = [FLAGS.nstride1, FLAGS.nstride2] # [2,2]
		pool_k_list =[FLAGS.nk1, FLAGS.nk2 ]  # [3, 3]
		#1 all-to-all hidden layer
		dense_list = [FLAGS.hidden1] # [300]
		keep_prob = FLAGS.dropout # 0.55
		numcell = data.numcell

		# define the network model 
		model = ConvNet(images_placeholder, 
			num_filter_list, filter_size_list, pool_stride_list, 
			pool_k_list, dense_list, keep_prob_placeholder,numcell)
		
		print("model shape is")
		print(model.output.get_shape())

		## Add to the Graph the Ops for loss calculation.
		loss = buildnet.loss(model.output, activity_placeholder)
		train_op = buildnet.training(loss, FLAGS.learning_rate)

		# Add the variable initializer Op.
		init = tf.global_variables_initializer()          

		# Create a session for running Ops on the Graph.
		sess = tf.Session()
		# Run the Op to initializactivitytraine the variables.
		sess.run(init)

		# setting up recording training progress
		steplist = []
		evallist = []
		trainlist = []
		earlystoplist = []
		rmeanlist = []
		rcelllist = []
    
		## Start the training loop.
		lossearlystopmin = 10e6 # large dummy value use for finding the minimum loss in the early stop data
		for step in xrange(FLAGS.max_steps):
			start_time = time.time()
			batchindex = np.random.permutation(numtrain)[0:numbatch]
			xtrainbatch = data.xtrain[batchindex ,:,:,:]
			ytrainbatch = data.ytrain[batchindex ,:]
      
			feed_dict={images_placeholder: xtrainbatch, activity_placeholder: ytrainbatch, keep_prob_placeholder: keep_prob}
			_, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)

			FVE = 1-loss_value/lossbaseline
			duration = time.time() - start_time
			# print progress.
			if step % 50 == 0:
				## Print status
				
				print('Step %d: loss = %.4f; FVE = %5.2f (%.3f sec)' % (step, loss_value, FVE, duration))
      			## save and evaluate the model 
			if (step - 1) % 200 == 0 or (step + 1) == FLAGS.max_steps or step == 1:
				## evaluate and save progress 
				
				steplist.append(step) # list of training steps
				## Evaluate against the training set.
				feed_dict={images_placeholder: data.xtrain, activity_placeholder: data.ytrain, keep_prob_placeholder:1}
				losstrain = sess.run(loss, feed_dict=feed_dict)
				trainlist.append(losstrain) # list of loss on training set
				## Evaluate against the eval set.
				
				feed_dict={images_placeholder: data.xeval, activity_placeholder: data.yeval, keep_prob_placeholder:1}
				losseval = sess.run(loss, feed_dict=feed_dict)
				evallist.append(losseval) # list of loss on eval set
				3#computting r eval on eval set
				actpredict_eval = sess.run(model.output, feed_dict=feed_dict)
				reval= np.zeros(data.numcell)
				for icell in range(data.numcell):
					reval[icell], peval = pearsonr(actpredict_eval[:,icell], data.yeval[:,icell])
					if np.isnan(reval[icell]):
						reval[icell] = 0
				rcelllist.append(reval) # list of r eval on eval set
				revalmean = np.mean(reval)
				rmeanlist.append(revalmean) # list of  mean r eval on eval set
				## Evaluate againts early stop data
				feed_dict={images_placeholder: data.xstop, activity_placeholder: data.ystop, keep_prob_placeholder:1}
				lossearlystop = sess.run(loss, feed_dict=feed_dict)
				earlystoplist.append(lossearlystop) # list of loss on early stop set

				## plot and save training
				if FLAGS.savetraining:
					mansavefig(trainlist, earlystoplist, evallist, rmeanlist, steplist, lossbaseline)

				## Finding the minumum loss on the early stop data set
				## check if new early stop min. If so, treat as best trained nextwork 
				if lossearlystop < lossearlystopmin:  # check if early stop is new min 
					lossearlystopmin =  lossearlystop # save loss as new minumum loss on the early stop data set
					FVEeval = 1-losseval/lossbaseline
					lossevalmin=losseval
					print("early stop")
					print("perfornace is")
					print('squared (loss) = %.4f; FVE = %5.3f' % (losseval,FVEeval))
					
					# compute r val again. Could use above values. 
					rval= np.zeros(data.numcell)
					feed_dict={images_placeholder: data.xeval, activity_placeholder: data.yeval, keep_prob_placeholder:1}
					actpredict_eval = sess.run(model.output, feed_dict=feed_dict)
					rval= np.zeros(data.numcell)
					for icell in range(data.numcell):
						rval[icell], pval = pearsonr(actpredict_eval[:,icell], data.yeval[:,icell])
						if np.isnan(rval[icell]):
							rval[icell] = 0
					rmean = np.mean(rval)
					print('r = %5.3f' % (rmean))
					print("more training!")
					if FLAGS.savenetwork:
						network_save(step) #save the parameters of network
					if FLAGS.save:
						plotandsaver(rval,step) #save the performance of network 

		print("Final results")
		print("Best perforance ")
		print('r = %5.3f +- %5.3f;  squared (loss) = %.4f;  FVE = %5.3f' % (rmean, np.std(rval)/np.sqrt(data.numcell), losseval, FVEeval))
		print("eval var is") 
		print(np.mean(evalvar)) 
		rerror = np.std(rval)/np.sqrt(data.numcell)   
        
        
def baseline_error(): 
	# make the baseline predictor
	meanactdum = np.sum(data.yeval, axis=0)/data.numtrials
	meanactdum  = np.mean(data.yeval, axis=0)
	
	meanact =  np.reshape(meanactdum, [1,meanactdum.size])
	meanpredict = np.repeat(meanact,data.numeval, axis=0)
	
	#make placeholdfers for baseline 
	y_ = tf.placeholder(tf.float32, shape=(None,data.numcell))
	meanpredict_placeholder = tf.placeholder(tf.float32, shape=(None,data.numcell))
	loss_baseline = buildnet.loss(meanpredict_placeholder,y_)
	loss_baseline_percell = buildnet.losspercell(meanpredict_placeholder,y_)
	
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	
	# Evaluate the baseline.
	feed_dict={meanpredict_placeholder: meanpredict, y_: data.yeval}
	loss_baseline_eval = sess.run(loss_baseline, feed_dict=feed_dict)
	loss_baseline_eval_percell = sess.run(loss_baseline_percell, feed_dict=feed_dict)
	#print and return baseline loss
	print('')
	print('Eval data baseline loss = %.4f' % (loss_baseline_eval))
	print("varience of eval")
	vare = np.var(data.yeval, axis = 0)
	print(np.shape(vare) )
	print(np.mean(vare))
	manvare = np.mean(np.mean(np.square(meanpredict-data.yeval)))
	print("man calc varience/loss of eval")
	print(manvare)
	return loss_baseline_eval, loss_baseline_eval_percell


def plotandsaver(rval,step):
	figN, axN = plt.subplots()
	bar_width = 0.40
	xvals = np.arange(len(rval))+1-bar_width/2
	axN.bar(xvals, rval)
	axN.set_xlabel('cell', color='k')
	axN.set_ylabel('r', color='k')
	plt.savefig(pearsonsavedir)
	traintrials = data.traintrials
	earlystoptrials = data.earlystoptrials
	evaltrials = data.evaltrials
	np.save(pearsonsavedatadir,[rval,step,traintrials, earlystoptrials, evaltrials, FLAGS])
 
  
def network_save(step):
	print('saving network as: ' + savenetworkname)
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	#manually get all of the network parameters
	with tf.variable_scope('conv1', reuse = True) as scope:
		weights1 = tf.get_variable('W')
		WC1 = sess.run(weights1)
		biases1 = tf.get_variable('b')
		BC1 =sess.run(biases1)
	with tf.variable_scope('conv2', reuse = True) as scope:
		weights2 = tf.get_variable('W')
		WC2 = sess.run(weights2)
		biases2 = tf.get_variable('b')
		BC2 =sess.run(biases2)
	with tf.variable_scope('dense1', reuse = True) as scope:
		weights3 = tf.get_variable('W')
		WH3 = sess.run(weights3)
		biases3 = tf.get_variable('b')
		BH3 = sess.run(biases3)
	with tf.variable_scope('linear', reuse = True) as scope:
		weights4 = tf.get_variable('W')
		WL4 = sess.run(weights4)
		biases4 = tf.get_variable('b')
		BL4 = sess.run(biases4)    
	np.save(savenetworkname,[WC1, BC1, WC2, BC2, WH3, BH3, WL4, BL4, step])


def mansavefig(trainlist, earlystoplist, evallist, rlist, step, lossbaseline):
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
	#axA.annotate('Image decimation: %d ' % (decimater), xy=(textx, 0.35*deltay+ymin))
	plt.legend(loc=4)
	axB = axA.twinx()
	axB.plot(np.asarray(step), np.asarray(rlist), 'k')
	axB.set_ylim([-0.1,1])
	axB.set_ylabel('r', color='k')
	axB.tick_params('y', colors='k')
	figA.tight_layout()
	plt.savefig(savefigname)
	
	##saving training parameters
	traintrials = data.traintrials
	earlystoptrials = data.earlystoptrials
	evaltrials = data.evaltrials
	np.save(savedirname,[FLAGS, trainlist, earlystoplist, evallist, rlist,
		step, lossbaseline, traintrials, earlystoptrials, evaltrials])
              

def main(_):
  lossbaseline, lossbaselinenueron  = baseline_error()
  run_training(lossbaseline)


# run main  
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  