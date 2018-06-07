import pytest
import tensorflow as tf
import numpy as np
import os

from src.buildnet import LnonL
from src.utils import DataLoader
from runnet import filearray

class MockParser():
    def __init__(self,**kwargs):
        for k,v in kwargs.items():
            setattr(self,k,v)

def mock_flags():
    data_dir = os.path.join(os.getcwd(),'data')

    parser = MockParser(
        data_dir=data_dir,
        trainingset_frac=2.0/3,
        earlystop_frac=1.0/7,
        )

    return parser

def test_LnonL():

    FLAGS = mock_flags()
    filename = filearray[1]
    filepath = os.path.join(FLAGS.data_dir,filename)
    data = DataLoader([filepath],FLAGS)
    assert isinstance(data.numcell,int)
    assert isinstance(data.images,np.ndarray)
    lin_layer = LnonL(data.images,ncell=data.numcell)

    return lin_layer

# def test_LNLN():