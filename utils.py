import numpy as np
import scipy.io as sio

class DataLoader(object):
	
	def __init__(self, filenamelist,FLAGS=None):	
		## load the data from files in 'filenamelist'
		## creats an object with trianing, early stop and evaluation formatted data 
		## update filenames in "filenamelist" and the argument in " = mat_contents['argument']"
		numfiles = len(filenamelist)

		print("loading %d file(s)" % (numfiles))
		for fn in filenamelist:
			print(fn)

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
