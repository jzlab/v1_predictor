import numpy as np
import tensorflow as tf

from runnet import baseline_error, run_training
from src.buildnet import LNLN
from .test_buildnet import MockParser, mock_data

def register_model(_):
    data = mock_data()
    imgs = data.images.astype(np.float32)
    model = LNLN(150,imgs,data.numcell)
    lossbaseline, lossbaselinenueron  = baseline_error()
    run_training(lossbaseline, lossbaselinenueron, model, dataset=data)

def test_LNLN(mock_data):
    tf.app.run(main=register_model)