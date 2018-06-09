import os
import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.ioff()

from src.utils import DataLoader
from src.parser import gen_parser
parser = gen_parser()
parser.add_argument(
  '--lnln_units',
  type=int,
  default=150,
  help='Number of units for hidden layer in LNLN model'
)

args,unparsed = parser.parse_known_args()

def load_data():
    data_dir = os.path.join( os.getcwd(), 'data')

    filename = filearray[args.fileindex]
    filepath = os.path.join( args.data_dir, filename )
    data = DataLoader( [filepath], args )

    return data

from src.runnet_fast import baseline_error, run_training, filearray
from src.buildnet import LNLN
from tqdm import tqdm as tqdm

def wrapper(_):
    data = load_data()
    imgs = data.images.astype(np.float32)
    model = LNLN(args.lnln_units, imgs, data.numcell)
    lossbaseline, lossbaselinenueron  = baseline_error()
    run_training(lossbaseline, lossbaselinenueron, model, dataset=data)

if __name__ == '__main__':
    parser = gen_parser()
    os.path.exists(os.path.join(os.getcwd(),'lnln_eval'))
    tf.app.run(main=register_model)
