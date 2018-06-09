import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.ioff()

from runnet import baseline_error, run_training
from src.buildnet import LNLN
from src.utils import DataLoader
from runnet import filearray

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

    filename = filearray[0]
    filepath = os.path.join(FLAGS.data_dir,filename)
    data = DataLoader([filepath],FLAGS)

    return data

def register_model(_):
    data = mock_data()
    imgs = data.images.astype(np.float32)
    model = LNLN(150,imgs,data.numcell)
    lossbaseline, lossbaselinenueron  = baseline_error()
    run_training(lossbaseline, lossbaselinenueron, model, dataset=data)


tf.app.run(main=register_model)
