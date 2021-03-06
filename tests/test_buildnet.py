import pytest
import tensorflow as tf
import numpy as np
import os

import keras.backend as K
from src.buildnet import LnonL, LNLN
from src.utils import DataLoader
from runnet import filearray

class MockParser():
    def __init__(self,**kwargs):
        for k,v in kwargs.items():
            setattr(self,k,v)

@pytest.fixture
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

def test_LnonL(mock_data):
    lin_layer = LnonL(mock_data.images,ncell=mock_data.numcell)

    return lin_layer

def test_LNLN(mock_data):
    imgs = mock_data.images.astype(np.float32)
    model = LNLN(250, imgs, mock_data.numcell)

    return model