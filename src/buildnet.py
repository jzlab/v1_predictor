#
#
# ==============================================================================

"""Builds multipule types of networks.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import numpy as np

class FlattenImg(object):
	def __init__(self):
		pass

	def __call__(self,images):
		imgflat = tf.contrib.layers.flatten(images)
		self.output = imgflat

		return self.output

class LNLN(object):
	def __init__(self, num_units, images, num_cells):
		self.num_units = num_units
		self.num_cells = num_cells
		self.compiled = False


	def compile(self):
		self.flatten = FlattenImg()

		self.dense_1 = tf.contrib.keras.layers.Dense(self.num_units,activation='relu')
		with tf.variable_scope('scaled_nonlinear'):
			print("building non linear layer")
			weights = tf.get_variable('weights',
				initializer = .1+tf.truncated_normal([self.num_units],
					stddev=0.02))
			self.biases = tf.get_variable('biases', initializer = tf.zeros([self.num_units]))
			self.diagweights1 = tf.diag(weights)
		self.dense_2 = tf.contrib.keras.layers.Dense(self.num_cells,activation='relu')

		self.compiled = True

	def __call__(self,images):
		assert self.compiled
		
		r = self.predict(images)
		self.output = r
		return r

	def predict(self,images):

		flat_img = self.flatten(images)
		x = self.dense_1(flat_img)
		sigmatmul = tf.matmul(x, self.diagweights1)
		nonlinear = tf.add(sigmatmul, self.biases)
		r = self.dense_2(nonlinear)

		self.output = r

		return r

class LnonL(object):
	"""
	A shallow linear non-linear network
	"""
	def __init__(
		self, images, ncell):
		
		## flatten input
		imgflat = tf.contrib.layers.flatten(images.astype(np.float32))
		imgflatshape = imgflat.get_shape()
		
		## linear layer
		with tf.variable_scope('linear'):
			print("building linear layer")
			numin = imgflatshape[1].value
			weights = tf.get_variable('weights',
				initializer = tf.truncated_normal([numin, ncell],
					stddev=0.1 / math.sqrt(float(numin))))
			biases = tf.get_variable('biases', initializer = tf.zeros([ncell]))
			#hidden2_drop = tf.nn.dropout(hidden2, keep_prob)
			#linear = tf.nn.relu(tf.add(tf.matmul(hidden2_drop, weights),biases))
			linear = tf.add(tf.matmul(imgflat, weights),biases)
			print("linear maping %d elemements to %d elements" % (numin, ncell)) 
		print(tf.shape(linear))
		## add parametrized sigmoid nonlinearity to output 
		with tf.variable_scope('nonlinear'):
			print("building non linear layer")
			weights = tf.get_variable('weights',
				initializer = .1+tf.truncated_normal([ncell],
					stddev=0.02))
			biases = tf.get_variable('biases', initializer = tf.zeros([ncell]))
			#hidden2_drop = tf.nn.dropout(hidden2, keep_prob)
			#linear = tf.nn.relu(tf.add(tf.matmul(hidden2_drop, weights),biases))
			print("linear")
			print(linear)
			print("weights")
			print(weights)
			diagweights = tf.diag(weights)
			print("diagweights")
			print(diagweights)
			#siglin = tf.sigmoid(linear)
			#nonlindum = tf.sigmoid(linear)
			nonlindum = tf.nn.relu(linear)
			sigmatmul = tf.matmul(nonlindum, diagweights)
			nonlinear = tf.add(sigmatmul ,biases)
			#print(" adding nonlinearity the %d outputs" % (ncell))
		
		self.output = nonlinear


class RConvNet(object):
	"""
	A Recurent convolutional nueral network to predict nueron firing rates due to images 
	"""
	def __init__(
		self, images, num_filter_list, filter_size_list, pool_stride_list,
		pool_k_list, dense_list, LSTM_list, keep_prob, ncell):
		#self, images,  img_pix, num_filter_list, filter_size_list, 
		#pool_stride_list, pool_k_list, dense_list, keep_prob, ncell):
		
		# ConvNet stage
		imgshape = images.get_shape()
		images_flatbatch = tf.reshape(images, [imgshape[0].value*imgshape[1].value, 
				imgshape[2].value, imgshape[3].value, imgshape[4].value])
		Convmodel = ConvLayers(images_flatbatch, num_filter_list, filter_size_list, 
				pool_stride_list, pool_k_list)
		# densely connected stage
		densein = tf.contrib.layers.flatten(Convmodel.output)
		Densemodel = DenseLayers(densein,dense_list)
		# LSTM stage
		RNNinput = tf.reshape(Densemodel.output,[imgshape[0].value,imgshape[1].value, -1]) 
		RNNmodel = RNNLayers(RNNinput, LSTM_list)
		LSTMout = RNNmodel.output
		# linear output of LSTM
		with tf.variable_scope('linear'):
			print("building linear layer")
			LSTMoutshape = LSTMout.get_shape()
			numelements = LSTMoutshape[2].value
			weights = tf.get_variable('weights',
				initializer = tf.truncated_normal([numelements, ncell],
					stddev=0.1 / math.sqrt(float(numelements))))
			biases = tf.get_variable('biases', initializer = tf.zeros([ncell]))
			LSTMoutFrontFlat = tf.reshape(LSTMout,[LSTMoutshape[0].value * LSTMoutshape[1].value, LSTMoutshape[2].value])
			linearFrontFlat = tf.add(tf.matmul(LSTMoutFrontFlat, weights),biases)
			linear = tf.reshape(linearFrontFlat, [LSTMoutshape[0].value, LSTMoutshape[1].value,ncell])
			print("linear maping %d elemements to %d elements" % (numelements, ncell))
		# full recurent conv net output
		self.output = linear
		
		# linear out of dense layer for pretraining
		with tf.variable_scope('linearpretrain'):
			print("building linear layer for pre training")
			Densemodelshape = Densemodel.output.get_shape()
			numelements = Densemodelshape[1].value
			weights = tf.get_variable('weights',
				initializer = tf.truncated_normal([numelements, ncell],
					stddev=0.1 / math.sqrt(float(numelements))))
			biases = tf.get_variable('biases', initializer = tf.zeros([ncell]))
			linearpretrain = tf.add(tf.matmul(Densemodel.output, weights),biases)
			print("linear maping %d elemements to %d elements" % (numelements, ncell))
		# output for pre training the conv
		self.pretrain = linearpretrain

class ConvNetDrop(object):
	"""
	A Convolutional nueral network to predict nueron firing rates due to images 
	"""
	def __init__(
		self, images, num_filter_list, filter_size_list, pool_stride_list,
		pool_k_list, dense_list, keep_prob, ncell):
		#self, images,  img_pix, num_filter_list, filter_size_list, 
		#pool_stride_list, pool_k_list, dense_list, keep_prob, ncell):
		
		# number of input channels
		#dum,dum,dum, imgchannels = tf.shape(images)
		print("start building network")
		dum,dum,dum, imgchannels = images.shape
		#num_channel_in = num_filter_list
		num_channel_in = []
		for i in range(len(filter_size_list)):
			if i == 0:
				#num_channel_in[i] = int(imgchannels)
				num_channel_in.append(int(imgchannels))
			else:
				#num_channel_in[i] = num_filter_list[i-1]
				num_channel_in.append(num_filter_list[i-1])
		# Create a convolution + maxpool layer for each filter size
		previous_layer = images
		for i in range(len(filter_size_list)):
			print("building layer conv%d" % (i+1))
			#with tf.name_scope("conv%d" % (i+1)):
			with tf.variable_scope("conv%d" % (i+1)):
				# Convolution Layer
				filter_shape = [filter_size_list[i], filter_size_list[i], num_channel_in[i], num_filter_list[i]]
				print("with %d filters shaped "  % (int(num_filter_list[i])))
				print(filter_shape)
				#W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
				W = tf.get_variable("W", initializer = tf.truncated_normal(filter_shape, stddev=0.1))
				#print('num_filter_list[i]')
				#print(num_filter_list[i])
				#b = tf.Variable(tf.constant(0.0, shape= [num_filter_list[i]]), name="b")
				b = tf.get_variable("b", initializer = tf.constant(0.0, shape= [num_filter_list[i]]))
				conv = tf.nn.conv2d(
					previous_layer,
					W,
					strides=[1, 1, 1, 1],
					padding="VALID",
					name="conv")
				# Apply nonlinearity
				h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
				# Maxpooling over the outputs
				pooled = tf.nn.max_pool(
					h,
					ksize=[1, pool_k_list[i], pool_k_list[i], 1],
					strides=[1, pool_stride_list[i], pool_stride_list[i], 1],
					padding='VALID',
					name="pool")
				current_conv_layer = pooled
				print("curret layer is ")
				print(current_conv_layer)
			previous_layer = current_conv_layer
		# flatted output of the conv-maxpool block
		convout_flat = tf.contrib.layers.flatten(current_conv_layer) 
		
		# create the densely connected block
		previous_layer = convout_flat
		for i in range(len(dense_list)):
			print("building layer dense%d" % (i+1))
			with tf.variable_scope("dense%d" % (i+1)):
				dim = previous_layer.get_shape()[1].value
				print("maping %d elemements to %d elements" % (dim, dense_list[i]))
				W = tf.get_variable('W',
					initializer=tf.truncated_normal([dim, dense_list[i]],
						stddev=0.1 / math.sqrt(float(dim))))
				b = tf.get_variable('b', initializer = tf.zeros([dense_list[i]]))
				current_dense_layer = tf.nn.relu(tf.add(tf.matmul(previous_layer, W),b))
			previous_layer = current_dense_layer
		denseout = previous_layer
		
		# linear
		with tf.variable_scope('linear'):
			print("building linear layer")
			dim = denseout.get_shape()[1].value
			weights = tf.get_variable('W',
				initializer = tf.truncated_normal([dim, ncell],
					stddev=0.1 / math.sqrt(float(dim))))
			biases = tf.get_variable('b', initializer = tf.zeros([ncell]))
			denseout_drop = tf.nn.dropout(denseout, keep_prob)
			linear = tf.nn.relu(tf.add(tf.matmul(denseout_drop, weights),biases))
			#linear = tf.nn.relu(tf.add(tf.matmul(denseout, weights),biases))
			print("linear maping %d elemements to %d elements" % (dim, ncell))
		self.output = linear
		print("Finished building network")



class ConvNet(object):
	"""
	A Convolutional nueral network to predict nueron firing rates due to images 
	"""
	def __init__(
		self, images, num_filter_list, filter_size_list, pool_stride_list,
		pool_k_list, dense_list, keep_prob, ncell):
		#self, images,  img_pix, num_filter_list, filter_size_list, 
		#pool_stride_list, pool_k_list, dense_list, keep_prob, ncell):
		
		# number of input channels
		#dum,dum,dum, imgchannels = tf.shape(images)
		print("start building network")
		dum,dum,dum, imgchannels = images.shape
		#num_channel_in = num_filter_list
		num_channel_in = []
		for i in range(len(filter_size_list)):
			if i == 0:
				#num_channel_in[i] = int(imgchannels)
				num_channel_in.append(int(imgchannels))
			else:
				#num_channel_in[i] = num_filter_list[i-1]
				num_channel_in.append(num_filter_list[i-1])
		# Create a convolution + maxpool layer for each filter size
		previous_layer = images
		for i in range(len(filter_size_list)):
			print("building layer conv%d" % (i+1))
			#with tf.name_scope("conv%d" % (i+1)):
			with tf.variable_scope("conv%d" % (i+1)):
				# Convolution Layer
				filter_shape = [filter_size_list[i], filter_size_list[i], num_channel_in[i], num_filter_list[i]]
				print("with %d filters shaped "  % (int(num_filter_list[i])))
				print(filter_shape)
				#W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
				W = tf.get_variable("W", initializer = tf.truncated_normal(filter_shape, stddev=0.1))
				#print('num_filter_list[i]')
				#print(num_filter_list[i])
				#b = tf.Variable(tf.constant(0.0, shape= [num_filter_list[i]]), name="b")
				b = tf.get_variable("b", initializer = tf.constant(0.0, shape= [num_filter_list[i]]))
				conv = tf.nn.conv2d(
					previous_layer,
					W,
					strides=[1, 1, 1, 1],
					padding="VALID",
					name="conv")
				# Apply nonlinearity
				h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
				# Maxpooling over the outputs
				pooled = tf.nn.max_pool(
					h,
					ksize=[1, pool_k_list[i], pool_k_list[i], 1],
					strides=[1, pool_stride_list[i], pool_stride_list[i], 1],
					padding='VALID',
					name="pool")
				current_conv_layer = pooled
				print("curret layer is ")
				print(current_conv_layer)
			previous_layer = current_conv_layer
		# flatted output of the conv-maxpool block
		convout_flat = tf.contrib.layers.flatten(current_conv_layer) 
		
		# create the densely connected block
		previous_layer = convout_flat
		for i in range(len(dense_list)):
			print("building layer dense%d" % (i+1))
			with tf.variable_scope("dense%d" % (i+1)):
				dim = previous_layer.get_shape()[1].value
				print("maping %d elemements to %d elements" % (dim, dense_list[i]))
				W = tf.get_variable('W',
					initializer=tf.truncated_normal([dim, dense_list[i]],
						stddev=0.1 / math.sqrt(float(dim))))
				b = tf.get_variable('b', initializer = tf.zeros([dense_list[i]]))
				current_dense_layer = tf.nn.relu(tf.add(tf.matmul(previous_layer, W),b))
			previous_layer = current_dense_layer
		denseout = previous_layer
		
		# linear
		with tf.variable_scope('linear'):
			print("building linear layer")
			dim = denseout.get_shape()[1].value
			weights = tf.get_variable('W',
				initializer = tf.truncated_normal([dim, ncell],
					stddev=0.1 / math.sqrt(float(dim))))
			biases = tf.get_variable('b', initializer = tf.zeros([ncell]))
			#hidden2_drop = tf.nn.dropout(hidden2, keep_prob)
			#linear = tf.nn.relu(tf.add(tf.matmul(hidden2_drop, weights),biases))
			linear = tf.nn.relu(tf.add(tf.matmul(denseout, weights),biases))
			print("linear maping %d elemements to %d elements" % (dim, ncell))
		self.output = linear
		print("Finished building network")


class ConvLayers(object):
	"""
	A Convolutional block of layers 
	"""
	def __init__(
		self, images, num_filter_list, filter_size_list, pool_stride_list,
		pool_k_list):
		
		# number of input channels
		#dum,dum,dum, imgchannels = tf.shape(images)
		print("start building network")
		dum,dum,dum, imgchannels = images.shape
		#num_channel_in = num_filter_list
		num_channel_in = []
		for i in range(len(filter_size_list)):
			if i == 0:
				#num_channel_in[i] = int(imgchannels)
				num_channel_in.append(int(imgchannels))
			else:
				#num_channel_in[i] = num_filter_list[i-1]
				num_channel_in.append(num_filter_list[i-1])
		# Create a convolution + maxpool layer for each filter size
		previous_layer = images
		for i in range(len(filter_size_list)):
			print("building layer conv%d" % (i+1))
			with tf.name_scope("conv%d" % (i+1)):
				# Convolution Layer
				filter_shape = [filter_size_list[i], filter_size_list[i], num_channel_in[i], num_filter_list[i]]
				print("with %d filters shaped "  % (int(num_filter_list[i])))
				print(filter_shape)
				W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
				#print('num_filter_list[i]')
				#print(num_filter_list[i])
				b = tf.Variable(tf.constant(0.0, shape= [num_filter_list[i]]), name="b")
				conv = tf.nn.conv2d(
					previous_layer,
					W,
					strides=[1, 1, 1, 1],
					padding="VALID",
					name="conv")
				# Apply nonlinearity
				h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
				# Maxpooling over the outputs
				pooled = tf.nn.max_pool(
					h,
					ksize=[1, pool_k_list[i], pool_k_list[i], 1],
					strides=[1, pool_stride_list[i], pool_stride_list[i], 1],
					padding='VALID',
					name="pool")
				current_conv_layer = pooled
				print("curret layer is ")
				print(current_conv_layer)
			previous_layer = current_conv_layer
		# flatted output of the conv-maxpool block
		convout=current_conv_layer
		convout_flat = tf.contrib.layers.flatten(current_conv_layer) 
		
		self.output = convout
		print("Finished building convolutional layers")

class DenseLayers(object):
	"""
	A densely connected block of layers 
	"""
	def __init__(self, input, dense_list):
		#self, images,  img_pix, num_filter_list, filter_size_list, 
		#pool_stride_list, pool_k_list, dense_list, keep_prob, ncell):
		
		# number of input channels
		#dum,dum,dum, imgchannels = tf.shape(images)
		print("start building densly connected layers")
		#dum,dum,dum, imgchannels = input.shape
		#num_channel_in = num_filter_list
		#input_flat = tf.contrib.layers.flatten(input) 
		
		# create the densely connected block
		#previous_layer = input_flat
		previous_layer = input
		for i in range(len(dense_list)):
			print("building layer dense%d" % (i+1))
			with tf.variable_scope("dense%d" % (i+1)):
				dim = previous_layer.get_shape()[1].value
				print("maping %d elemements to %d elements" % (dim, dense_list[i]))
				weights = tf.get_variable('weights',
					initializer=tf.truncated_normal([dim, dense_list[i]],
						stddev=0.1 / math.sqrt(float(dim))))
				biases = tf.get_variable('biases', initializer = tf.zeros([dense_list[i]]))
				current_dense_layer = tf.nn.relu(tf.add(tf.matmul(previous_layer, weights),biases))
			previous_layer = current_dense_layer
		denseout = previous_layer
		self.output = denseout
		print("Finished building densely connected layers")

class RNNLayers(object):
	"""
	A block or recurrent layers
	"""
	def __init__(
		self, input, LSTM_list):
		
		print("start building recurrent layers")
		
		## flatten input
		inputshape = input.get_shape()
		#input_flat = tf.contrib.layers.flatten(images)
		input_flat = tf.reshape(input,[inputshape[0].value,inputshape[1].value,-1])
		#input_flat = tf.reshape(images,[imgshape[0],imgshape[1],imgshape[2]*imgshape[3]*imgshape[4]])
		
		## start stack of LSTM layers
		previous_layer = input_flat
		for i in range(len(LSTM_list)):
			print("building layer LSTM%d" % (i+1))
			print("with input shape: ")
			print(previous_layer.get_shape())
			with tf.variable_scope("LSTM%d" % (i+1)):
				dim = previous_layer.get_shape()[1].value
				print("maping %d elemements to %d LSTM(s)" % (dim, LSTM_list[i]))
				cell = tf.contrib.rnn.BasicLSTMCell(LSTM_list[i], forget_bias=1.0, state_is_tuple=True)
				output, _ = tf.nn.dynamic_rnn(cell, previous_layer, dtype=tf.float32)
			previous_layer = output
		LSTMout = output
		## set outputs
		self.output = LSTMout
		print("Finished building recurrent layers")


class simpleRNN(object):
	"""
	A simple recurrent network
	"""
	def __init__(
		self, images, LSTM_list, ncell):
		
		print("start building simple RNN")
		
		## flatten input
		imgshape = images.get_shape()
		#input_flat = tf.contrib.layers.flatten(images)
		input_flat = tf.reshape(images,[imgshape[0].value,imgshape[1].value,-1])
		#input_flat = tf.reshape(images,[imgshape[0],imgshape[1],imgshape[2]*imgshape[3]*imgshape[4]])
		
		## start stack of LSTM layers
		previous_layer = input_flat
		for i in range(len(LSTM_list)):
			print("building layer LSTM%d" % (i+1))
			print("with input shape: ")
			print(previous_layer.get_shape())
			with tf.variable_scope("LSTM%d" % (i+1)):
				dim = previous_layer.get_shape()[1].value
				print("maping %d elemements to %d LSTM(s)" % (dim, LSTM_list[i]))
				cell = tf.contrib.rnn.BasicLSTMCell(LSTM_list[i], forget_bias=1.0, state_is_tuple=True)
				output, _ = tf.nn.dynamic_rnn(cell, previous_layer, dtype=tf.float32)
			previous_layer = output
		LSTMout = output
		
		## linear layer 
		with tf.variable_scope('linear'):
			print("building linear layer")
			LSTMoutshape = LSTMout.get_shape()
			numelements = LSTMoutshape[2].value
			weights = tf.get_variable('weights',
				initializer = tf.truncated_normal([numelements, ncell],
					stddev=0.1 / math.sqrt(float(dim))))
			biases = tf.get_variable('biases', initializer = tf.zeros([ncell]))
			#hidden2_drop = tf.nn.dropout(hidden2, keep_prob)
			#linear = tf.nn.relu(tf.add(tf.matmul(hidden2_drop, weights),biases))
			LSTMoutFrontFlat = tf.reshape(LSTMout,[LSTMoutshape[0].value * LSTMoutshape[1].value, LSTMoutshape[2].value])
			linearFrontFlat = tf.add(tf.matmul(LSTMoutFrontFlat, weights),biases)
			linear = tf.reshape(linearFrontFlat, [LSTMoutshape[0].value, LSTMoutshape[1].value,ncell])
			print("linear maping %d elemements to %d elements" % (numelements, ncell))
		
		## set outputs
		self.output = linear
		print("Finished building network")

def loss(linear, y_):
	"""Calculates the least squre loss from the measured and predicted activity .
	
	Args:
	
	
	
	Returns:
	loss: Loss tensor of type float.
	"""
	return tf.reduce_mean(tf.square(linear - y_), name = 'least_squares')


def losspercell(linear, y_):
	"""Calculates the least squre loss from the measured 
	and predicted activity for each cell.

	#Args:
	#l
	#l
	
	#Returns:
	#loss: Loss tensor of type float size num cells.
	#"""
	##labels = tf.to_int64(labels)
	##cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
	##    labels=labels, logits=logits, name='xentropy')
	return tf.reduce_mean(tf.square(linear - y_),0, name = 'least_squares')#tf.reduce_mean(cross_entropy, name='xentropy_mean')


def lossloglike(linear, y_):
	"""Calculates the least squre loss from the measured 
	and predicted activity for each cell.

	#Args:
	#l
	#l
	
	#Returns:
	#loss: Loss tensor of type float size num cells.
	#"""
	##labels = tf.to_int64(labels)
	##cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
	##    labels=labels, logits=logits, name='xentropy')
	epsilon = 1e-6
	lossvector = linear - tf.multiply(y_, tf.log(linear + epsilon))
	return tf.reduce_mean(lossvector, name = 'log_like')
	#return tf.reduce_mean(tf.square(linear - y_),0, name = 'least_squares')#tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss, learning_rate):
	"""Sets up the training Ops.
	Creates an optimizer and applies the gradients to all trainable variables.
	Args:
	loss: Loss tensor, from loss().
	learning_rate: The learning rate to use for gradient descent.
	Returns:
	train_op: The Op for training.
	"""
	## Create the gradient descent optimizer with the given learning rate.
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	## Create a variable to track the global step.
	global_step = tf.Variable(0, name='global_step', trainable=False)
	## Use the optimizer to apply the gradients that minimize the loss
	## (and also increment the global step counter) as a single training step.
	train_op = optimizer.minimize(loss, global_step=global_step)
	return train_op

