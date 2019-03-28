from .parser import gen_parser()
from keras.layers import Conv2D,Dense,Input

parser = gen_parser()
FLAGS,unparsed = parser.parse_known_args()

# network hyper-parameters as arrays
#2 conv layers
if FLAGS.numconvlayer == 1:
        num_filter_list = [FLAGS.conv1]
        filter_size_list = [FLAGS.conv1size]
        pool_stride_list = [FLAGS.nstride1]
        pool_k_list =[FLAGS.nk1]
elif FLAGS.numconvlayer == 2:
        num_filter_list = [FLAGS.conv1, FLAGS.conv2] # [16,32]
        filter_size_list = [FLAGS.conv1size, FLAGS.conv2size] # [7,7]
        pool_stride_list = [FLAGS.nstride1, FLAGS.nstride2] # [2,2]
        pool_k_list =[FLAGS.nk1, FLAGS.nk2 ]  # [3, 3]
elif FLAGS.numconvlayer == 3:
        num_filter_list = [FLAGS.conv1, FLAGS.conv2, FLAGS.conv2] # [16,32]
        filter_size_list = [FLAGS.conv1size, FLAGS.conv2size, FLAGS.conv2size] # [7,7]
        pool_stride_list = [FLAGS.nstride1, FLAGS.nstride2, FLAGS.nstride2] # [2,2]
        pool_k_list =[FLAGS.nk1, FLAGS.nk2, FLAGS.nk2]  # [3, 3]

#1 all-to-all hidden layer
dense_list = [FLAGS.hidden1] # [300]
keep_prob = FLAGS.dropout # 0.55
# Attempt to make port wk's class definitions to keras to make easier to work with
class ConvNetDrop(object):
    def __init__(self, images, num_filter_list, filter_size_list, pool_stride_list):
        self.num_filter_list = num_filter_list
        self.input = Input(shape=(28,28,1),name='image_input')

    def build(self,x):
        net = Conv2D(units,kernel_initializer='truncated_normal',bias_initializer='zeros')(x)
