import os
import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.ioff()

from src.runnet_fast import baseline_error, run_training, filearray
from src.buildnet import LNLN
from src.utils import DataLoader
from tqdm import tqdm as tqdm

NUM_TRIALS = 7
NUM_UNITS = 150 # Default value
if len(sys.argv)>1:
    NUM_UNITS = int(sys.argv[1])

class MockParser():
    def __init__(self,**kwargs):
        for k,v in kwargs.items():
            setattr(self,k,v)

def mock_data():
    data_dir = os.path.join(os.getcwd(),'data')

    FLAGS = MockParser(
        data_dir=data_dir,
        trainingset_frac=2.0/3,
        earlystop_frac=1.0/7,
        )

    filename = filearray[1]
    filepath = os.path.join(FLAGS.data_dir,filename)
    data = DataLoader([filepath],FLAGS)

    return data

def register_model(_):
    data = mock_data()
    imgs = data.images.astype(np.float32)
    model = LNLN(NUM_UNITS,imgs,data.numcell)
    lossbaseline, lossbaselinenueron  = baseline_error()
    run_training(lossbaseline, lossbaselinenueron, model, dataset=data)


if __name__ == '__main__':
    os.path.exists(os.path.join(os.getcwd(),'lnln_hyperopt',str(150)))
    for _ in tqdm(np.arange(NUM_TRIALS)):
        tf.app.run(main=register_model)
