import sys
import pytest

import tensorflow as tf
import numpy as np

from runnet import run_training, baseline_error

def main(_):
    lossbaseline, lossbaselinenueron  = baseline_error()
    run_training(lossbaseline, lossbaselinenueron)

# def test_run_training():
#     tf.app.run(main=main, argv=[sys.argv[0]])
